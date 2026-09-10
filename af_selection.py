"""
af_selection.py
==================
THE HAND-EDITABLE ACQUISITION-FUNCTION RULE BOOK.

This is the one file you edit to tune the dynamic acquisition policy. Every
iteration of the BO loop, the workhorse (botorch_bo.py) hands us the current
state of the optimisation and asks: "which acquisition function should I use to
pick the next experiment?". `AFSelector.select()` answers that question by
walking a small decision tree of human-readable rules.

------------------------------------------------------------------------------
WHERE STATE LIVES  (three places, one job each)
------------------------------------------------------------------------------
  * `HeuristicConfig`      -- everything used to TUNE the rules (thresholds and
                              rolling-window lengths). Edit these to tune by hand.
  * `IterationHistory`     -- all metrics needed for the decision at the ITERATION
                              level (computed fresh each call: improvement (+rate),
                              uncertainty (level + running-max bands), normalised
                              nn_distance (+rate), incumbent f-stats, and the AF
                              that fired). One record per iteration; the list of
                              them is the plot/trace source.
  * `OptimisationHistory`  -- all metrics needed for the decision at the
                              OPTIMISATION level (cross-iteration memory: the
                              rolling windows, running histories, the per-AF
                              trust/patience trackers, and the previous incumbent).

`AFSelector` itself holds only `self.config`, `self.history` (an
OptimisationHistory), and the fixed campaign params (total_budget, n_init,
problem_dim).

------------------------------------------------------------------------------
FRAME / SIGN CONVENTION  (important, read once)
------------------------------------------------------------------------------
The objective is MAXIMISED directly (no sign flip). We track an incumbent
`state.f_best_so_far` (the best objective value observed, which should
go UP) alongside `state.f_worst_so_far/f_mean/f_std/f_latest`. `f_latest` is THIS
iteration's pick value (can be above/equal/below the incumbent);
`previous_f_best_so_far` (kept in history) is the incumbent as of the previous
iteration.

"Progress" is the fraction of the BO budget spent, EXCLUDING the random initial
points:  budget_fraction = (N - n_init) / (total_budget - n_init).

------------------------------------------------------------------------------
CALIBRATION NOTE (kept visible on purpose)
------------------------------------------------------------------------------
With the narrow diamgate lengthscale prior, fitted lengthscales sit around ~3
with a tiny std (~0.1). So the verbatim lengthscale thresholds
(mean < 0.5, std > 0.5*mean, min < 0.1*max) almost never fire. The
per-iteration rule trace in the HTML report makes this visible -- if a rule
never fires, its thresholds are mis-scaled for our setup and are the first thing
to retune.
"""

from __future__ import annotations
import os
import json
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from collections import deque
import numpy as np

# Named configs are archived here (only needed if you use save_config/load_config).
VERSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rule_versions")


# =============================================================================
# 1. TUNABLE THRESHOLDS  --  edit these, then re-run.
# =============================================================================
@dataclass
class HeuristicConfig:
    """Every knob the rule book uses. Edit the defaults to tune by hand.

    The defaults below are the "v0_verbatim" rules (a faithful transcription of
    the originally hand-written policy). Each field comment says what the knob
    controls and the direction of its effect.
    """

    name: str = "v0_verbatim"

    early_phase_end: float = 0.15  # smaller -> shorter MES/TS burst at the start
    final_exploit_start: float = 0.90  # smaller -> stop exploring & exploit earlier
    improve_fast_fraction: float = (
        0.5  # improvement_rate >= frac*max_improvement_rate -> "improving fast"
    )
    improve_threshold_fraction: float = (
        0.0  # incumbent gain must exceed frac*f_best_so_far to count as improving
    )
    improvement_window: int = (
        3  # sliding window length for improvement in f_best_so_far
    )
    uncertainty_lower_fraction: float = (
        0.3  # current_uncertainty < frac*max_uncertainty -> low (exploit); BORA value
    )
    uncertainty_upper_fraction: float = (
        0.5  # current_uncertainty > frac*max_uncertainty -> high (explore); BORA value
    )
    nn_distance_window: int = (
        3  # sliding window length for (N^(1/D)-normalised) nearest-neighbour distance
    )
    trust_window_size: int = 3  # sliding window length for per-AF trust scores
    rate_decay: float = (
        3.0  # trust score decay sharpness (higher -> a bad pick drops trust faster)
    )
    trust_penalty_threshold: float = (
        0.5  # trust below this -> a miss costs 2 patience instead of 1
    )
    patience_ceiling_factor: float = (
        2.0  # patience ceiling = round(factor * base_patience); raise for more buffer
    )
    improvement_flat_fraction: float = (
        0.1  # improvement_rate <= frac*max_improvement_rate -> "almost flat"
    )
    max_af_streak: int = (
        6  # same AF active >= this many iters + flat improvement -> force explore
    )
    explore_burst: int = (
        3  # length of a committed explore burst before returning to LogEI
    )
    stall_patience: int = (
        5  # LogEI iterations with NO incumbent improvement before an explore burst
    )


# =============================================================================
# 2. PER-ITERATION RECORD  --  all metrics for THIS iteration's decision.
# =============================================================================
@dataclass
class IterationHistory:
    """All metrics needed for the decision at the ITERATION level, computed fresh
    each call. One record is appended to OptimisationHistory.iterations per
    iteration; the list of them is exactly what the HTML report plots/tabulates."""

    progress: float = 0.0  # budget fraction spent (excl. random init)

    # improvement signal
    is_improving: bool = False
    improvement: float = 0.0
    improvement_rate: float = 0.0  # mean improvement per iteration over the window
    max_improvement_rate: float = 0.0  # running max of improvement_rate (the scale)

    # uncertainty signal (BORA-style: level vs running-max bands, no ratio/rate)
    current_uncertainty: float = (
        0.0  # mean GP posterior std over the fixed reference set
    )
    max_uncertainty: float = 0.0  # running max of current_uncertainty so far
    uncertainty_lower: float = (
        0.0  # = lower_fraction * max_uncertainty (below -> exploit)
    )
    uncertainty_upper: float = (
        0.0  # = upper_fraction * max_uncertainty (above -> explore)
    )

    # nearest-neighbour distance signal (normalised by the space-filling expectation)
    current_nn_distance: float = (
        0.0  # raw NN distance of the most recent pick (diagnostic)
    )
    nn_distance_normalised: float = 0.0  # raw * N^(1/D): stationary under space-filling
    nn_distance_rate: float = 0.0  # slope of the normalised nn_distance over the window

    # objective statistics (f_range scales the improve_fast threshold)
    f_best_so_far: float = 0.0
    f_worst_so_far: float = 0.0
    f_range: float = 0.0

    # decision outputs (filled in by select())
    af: Optional[str] = None
    rule_tag: str = ""
    reason: str = ""
    cfg: str = ""


def compute_af_patience(problem_dim: int) -> int:
    """Base patience (consecutive non-improving iterations tolerated before an AF
    is switched out) as a function of the problem dimension (i.e. the number of
    experimental parameters, NOT the feature-space dimension).
    """
    return max(1, math.ceil(max(problem_dim, 1) ** (1.0 / 3.0)))


class AFTrust:
    """Sliding-window trust score for an acquisition function, based on how good
    its recent picks were relative to the incumbent.
    0 = no trust, 1 = full trust."""

    def __init__(self, config: HeuristicConfig = HeuristicConfig()):
        # rolling window; left EMPTY so the first real outcome sets trust at once
        self._window = deque(maxlen=config.trust_window_size)
        self._rate_decay: float = config.rate_decay

    def update(self, y_current, y_previous_best):
        """Update the trust score from the latest pick. `y_current` is the pick's
        value; `y_previous_best` is the incumbent BEFORE this pick. delta > 0 ->
        the pick set a new record (full trust); delta <= 0 -> decay by how far
        below the incumbent the pick landed."""
        delta = y_current - y_previous_best
        if delta > 0:
            score = 1.0
        else:
            # decay proportional to how far below the incumbent the pick fell
            norm_delta = delta / (abs(y_previous_best) + 1e-6)
            score = 1.0 / (math.exp(-self._rate_decay * norm_delta))
        self._window.append(score)

    @property
    def trust(self):
        # the window is never read before the first update() in normal flow;
        # 1.0 is a neutral fallback only if it somehow is (treats unknown as good).
        if not self._window:
            return 1.0
        return sum(self._window) / len(self._window)


# =============================================================================
# 3. CROSS-ITERATION MEMORY  --  all metrics for OPTIMISATION-level decisions.
# =============================================================================
@dataclass
class OptimisationHistory:
    """All cross-iteration memory: the rolling windows, running histories, the
    per-AF trust/patience trackers, and the previous incumbent. Build one
    per campaign with `OptimisationHistory.create(config, problem_dim)` so the
    windows get their configured maxlen and the patience bounds get set."""

    base_patience: int = 1
    patience_bounds: List[int] = field(default_factory=lambda: [0, 2])

    # incumbent memory (only the lag is needed; state carries the current best)
    previous_f_best_so_far: Optional[float] = None
    is_first_call: bool = True

    # active-AF streak: consecutive iterations the SAME AF has been used (for the
    # "long flat streak -> force explore" override)
    last_af: Optional[str] = None
    current_af_streak: int = 0

    # committed explore-burst countdown (>0 => stay on the current explorer)
    explore_burst_remaining: int = 0

    # iterations since the incumbent last improved (the plateau / stall counter)
    iters_since_improvement: int = 0

    # uncertainty running max (BORA-style: bands are fractions of this)
    max_uncertainty: float = 0.0

    # improvement-rate running max (the "flat"/"fast" tests are fractions of this)
    max_improvement_rate: float = 0.0

    # rolling windows (maxlen set in create())
    improvement_window: deque = field(default_factory=deque)
    nn_distance_window: deque = field(default_factory=deque)

    # running history of the per-iteration records (the plot/trace source)
    iterations: List[IterationHistory] = field(default_factory=list)

    # per-AF trust / patience trackers
    af_trust: Dict[str, AFTrust] = field(default_factory=dict)
    af_patience: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def create(cls, config: HeuristicConfig, problem_dim: int) -> "OptimisationHistory":
        """Build a fresh history with windows sized from `config`, patience
        starting at the base and capped at factor*base."""
        base_patience = compute_af_patience(problem_dim)
        ceiling = max(
            base_patience, round(config.patience_ceiling_factor * base_patience)
        )
        history = cls(
            base_patience=base_patience,
            patience_bounds=[0, ceiling],
        )
        # seed the windows (improvement and normalised nn_distance start at 0)
        history.improvement_window = deque([0.0], maxlen=config.improvement_window)
        history.nn_distance_window = deque([0.0], maxlen=config.nn_distance_window)
        return history


def save_config(config: HeuristicConfig, directory: str = VERSIONS_DIR) -> str:
    """Archive a named config to rule_versions/<name>.json (optional convenience)."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{config.name}.json")
    with open(path, "w") as handle:
        json.dump(asdict(config), handle, indent=2)
    return path


def load_config(name: str, directory: str = VERSIONS_DIR) -> HeuristicConfig:
    """Load a previously-archived named config from rule_versions/<name>.json.
    Silently drops any keys that are not current HeuristicConfig fields."""
    with open(os.path.join(directory, f"{name}.json")) as handle:
        raw = json.load(handle)
    valid_fields = set(HeuristicConfig.__dataclass_fields__)
    cleaned = {key: value for key, value in raw.items() if key in valid_fields}
    return HeuristicConfig(**cleaned)


# =============================================================================
# 4. THE RULE BOOK
# =============================================================================
class AFSelector:
    """Stateful, per-iteration acquisition-function selector.

    Create one instance per BO campaign, then call `select(state, f_best_so_far)`
    once per iteration. All cross-iteration memory lives in `self.history` (an
    OptimisationHistory); all tuning knobs live in `self.config`.

    `select()` returns a 4-tuple:
        (acquisition_function, rule_tag, human_reason, diagnostics_dict)
    where `diagnostics_dict` carries every signal that drove the decision, so
    the HTML report can show exactly which rule fired at each iteration and why.
    """

    def __init__(
        self,
        total_budget: int = 50,
        n_init: int = 0,
        problem_dim: int = 1,  # number of experimental parameters
        config: Optional[HeuristicConfig] = None,
    ):
        self.total_budget = total_budget
        self.n_init = n_init
        self.problem_dim = problem_dim
        self.config = config or HeuristicConfig()

        # all cross-iteration memory lives here
        self.history = OptimisationHistory.create(self.config, problem_dim)

    # --- Internal helpers ---
    def _trust(self, af) -> AFTrust:
        """One trust tracker per AF."""
        if af not in self.history.af_trust:
            self.history.af_trust[af] = AFTrust(self.config)
        return self.history.af_trust[af]

    def _patience(self, af) -> int:
        """Patience tracker per AF (lazily initialised to the base patience)."""
        if af not in self.history.af_patience:
            self.history.af_patience[af] = self.history.base_patience
        return self.history.af_patience[af]

    def _retire_af(self, af) -> None:
        """Clean re-entry for an AF being switched out: clear its trust window and
        reset its patience to base, so a later re-selection is a fresh fair trial
        (arriving with the short burst leash, not judged on stale trust)."""
        self.history.af_trust[af] = AFTrust(self.config)
        self.history.af_patience[af] = self.history.base_patience

    # --- Main functions ---
    def trust_patience_select(self, state, pool_af: List[str]):
        """Assess the AF used last iteration and decide whether to switch.

        Returns ``(improvement, is_improving, switch_needed, candidate_pool)``:
          - improvement / is_improving: incumbent change since the previous call.
          - switch_needed: the active AF exhausted its patience.
          - candidate_pool: AFs viable for the next iteration (the allowed pool
            minus the exhausted AF when switching; ``[current_af]`` when staying).
            The metric dial in select() ranks these.
        Always returns the 4-tuple (no None path).
        """
        improvement = 0.0
        is_improving = False
        switch_needed = False
        candidate_pool = list(pool_af)

        if (
            not self.history.is_first_call
            and self.history.previous_f_best_so_far is not None
            and state.current_af
        ):
            previous_best = self.history.previous_f_best_so_far

            # incumbent improvement since the previous iteration
            delta_incumbent = state.f_best_so_far - previous_best
            improvement = max(delta_incumbent, 0.0)
            is_improving = delta_incumbent > max(
                self.config.improve_threshold_fraction * state.f_best_so_far, 1e-12
            )

            # trust for the AF just used: how good was its pick vs the PREVIOUS
            # best (updated first, then read to grade the patience penalty).
            self._trust(state.current_af).update(state.f_latest, previous_best)
            trust = self._trust(state.current_af).trust

            # --- patience as fuel (active AF only) ----------------------------
            # An improving pick refuels the active AF (+1); a miss burns it (-1,
            # or -2 if its recent track record is bad). Only the active AF moves:
            # a winning workhorse climbs toward the ceiling (a long, earned leash),
            # while every other AF sits at base -- so a newly-activated AF always
            # arrives with the short "burst" leash. Switch when the active AF runs
            # out of patience.
            def constrain_value(value):
                return max(
                    self.history.patience_bounds[0],
                    min(self.history.patience_bounds[1], value),
                )

            if is_improving:
                active_delta = 1
            else:
                active_delta = (
                    -1 if trust >= self.config.trust_penalty_threshold else -2
                )

            self.history.af_patience[state.current_af] = constrain_value(
                self._patience(state.current_af) + active_delta
            )

            # --- AF switch trigger: active AF ran out of patience -------------
            if self._patience(state.current_af) <= self.history.patience_bounds[0]:
                switch_needed = True
                candidate_pool = [af for af in pool_af if af != state.current_af]
                self._retire_af(
                    state.current_af
                )  # clear trust + reset patience to base
            else:
                candidate_pool = [state.current_af]

        return improvement, is_improving, switch_needed, candidate_pool

    def update_history(self, state, pool_af: List[str]):
        """Ingest the last outcome, update the cross-iteration memory, build the
        IterationHistory record for THIS iteration (the plot/trace source), and
        return the switch decision for select() to act on.

        Returns ``(record, switch_needed, candidate_pool)``.
        """

        # Compute the improvement in the last iteration, update patience and trust
        # in AFs, and make a decision on whether to switch AFs.
        improvement, is_improving, switch_needed, candidate_pool = (
            self.trust_patience_select(state, pool_af)
        )

        # track how long the same AF has been active (for the stuck override)
        if state.current_af is not None:
            if state.current_af == self.history.last_af:
                self.history.current_af_streak += 1
            else:
                self.history.current_af_streak = 1
            self.history.last_af = state.current_af

        # plateau counter: iterations since the incumbent last improved. Only count
        # while EXPLOITING (not during an explore burst), so LogEI gets a full
        # stall_patience runway after each burst.
        if is_improving:
            self.history.iters_since_improvement = 0
        elif self.history.explore_burst_remaining <= 0:
            self.history.iters_since_improvement += 1

        # --- improvement: average gain per iteration over the window ----------
        # "rate of improvement" = mean improvement per iteration (the LEVEL of
        # recent gains): steady solid gains read high, a grind of tiny gains reads
        # ~flat. (The slope/acceleration is NOT what the explore/exploit decisions
        # want -- constant improvement has zero slope but is clearly not "flat".)
        self.history.improvement_window.append(improvement)
        improvement_rate = sum(self.history.improvement_window) / len(
            self.history.improvement_window
        )
        self.history.max_improvement_rate = max(
            self.history.max_improvement_rate, improvement_rate
        )

        # --- uncertainty: BORA-style level vs running-max bands (no ratio/rate) -
        # current_uncertainty = mean GP posterior std over the FIXED reference set
        # (constant support, so comparable across iterations). Thresholds are
        # fractions of the running max: below lower -> model confident (exploit),
        # above upper -> uncertainty still high (explore).
        current_uncertainty = state.posterior_uncertainty
        self.history.max_uncertainty = max(
            self.history.max_uncertainty, current_uncertainty
        )
        uncertainty_lower = (
            self.config.uncertainty_lower_fraction * self.history.max_uncertainty
        )
        uncertainty_upper = (
            self.config.uncertainty_upper_fraction * self.history.max_uncertainty
        )

        # --- nearest-neighbour distance: normalise by the space-filling expectation
        # Raw min-distance to the sampled set shrinks as N grows for ANY strategy
        # (~N^(-1/D)), so multiply by N^(1/D) to get a quantity that is roughly
        # stationary under space-filling sampling. A FALLING normalised trend then
        # genuinely means clustering picks (over-exploiting). NaN guard for the
        # first pick (no neighbour yet).
        current_nn_distance = state.shortest_distance
        if current_nn_distance == current_nn_distance:  # not NaN
            nn_distance_normalised = current_nn_distance * (
                state.N ** (1.0 / max(state.D, 1))
            )
            self.history.nn_distance_window.append(nn_distance_normalised)
        else:
            nn_distance_normalised = float("nan")
        if len(self.history.nn_distance_window) >= 2:
            x = np.arange(len(self.history.nn_distance_window))
            y = np.array(self.history.nn_distance_window)
            slope, _ = np.polyfit(x, y, 1)
            nn_distance_rate = slope
        else:
            nn_distance_rate = 0.0

        # Budget fraction spent, excluding the random-init points.
        budget_fraction = (state.N - self.n_init) / max(
            self.total_budget - self.n_init, 1
        )
        objective_range = max(state.f_best_so_far - state.f_worst_so_far, 1e-12)

        # --- assemble and store the per-iteration record ----------------------
        record = IterationHistory(
            progress=budget_fraction,
            is_improving=bool(is_improving),
            improvement=improvement,
            improvement_rate=improvement_rate,
            max_improvement_rate=self.history.max_improvement_rate,
            current_uncertainty=current_uncertainty,
            max_uncertainty=self.history.max_uncertainty,
            uncertainty_lower=uncertainty_lower,
            uncertainty_upper=uncertainty_upper,
            current_nn_distance=current_nn_distance,
            nn_distance_normalised=nn_distance_normalised,
            nn_distance_rate=nn_distance_rate,
            f_best_so_far=state.f_best_so_far,
            f_worst_so_far=state.f_worst_so_far,
            f_range=objective_range,
            cfg=self.config.name,
        )
        self.history.iterations.append(record)

        # advance the cross-iteration cursors
        self.history.is_first_call = False
        self.history.previous_f_best_so_far = state.f_best_so_far
        return record, switch_needed, candidate_pool

    def select(self, state, f_best_so_far: float):
        """Decide the acquisition function for THIS iteration.

        Phase by budget: EARLY {MES,TS} -> MID -> LATE {PosMean}. In MID, LogEI is
        the exploit workhorse (governed by the trust/patience engine); when it
        stalls or we are grinding (flat gains + clustering / overstay), we run a
        short COMMITTED explore burst with a single explorer (UCB if very uncertain,
        else MES) and then return to LogEI -- no per-iteration thrash.
        Returns ``(acquisition_function, rule_tag, reason, diagnostics_dict)``.
        """
        cfg = self.config

        # --- phase + allowed pool --------------------------------------------
        budget_fraction = (state.N - self.n_init) / max(
            self.total_budget - self.n_init, 1
        )
        if budget_fraction >= cfg.final_exploit_start:
            phase, pool = "late", ["PosMean"]
        elif budget_fraction < cfg.early_phase_end:
            phase, pool = "early", ["TS", "MES"]
        else:
            phase, pool = "mid", ["LogEI", "UCB", "TS", "MES"]

        # --- ingest the previous outcome (updates trust/patience/metrics) -----
        was_first_call = self.history.is_first_call
        record, switch_needed, candidate_pool = self.update_history(state, pool)

        # Finalise: stamp the decision onto the record and return the 4-tuple.
        def emit(af, rule_tag, reason):
            record.af = af
            record.rule_tag = rule_tag
            record.reason = reason
            diagnostics = asdict(record)
            diagnostics.pop("af", None)  # choose() re-adds 'af' itself
            diagnostics["rule"] = rule_tag  # bo_report groups the trace by 'rule'
            return af, rule_tag, reason, diagnostics

        # --- LATE: pure exploitation -----------------------------------------
        if phase == "late":
            return emit("PosMean", "late_exploit", "final budget -> PosMean")

        # --- FIRST CALL: seed with Thompson sampling -------------------------
        if was_first_call:
            return emit("TS", "seed", "first sampling -> TS")

        # --- EARLY: MES/TS via the trust/patience engine ---------------------
        if phase == "early":
            if switch_needed and candidate_pool:
                return emit(
                    candidate_pool[0], "early_switch", "early stagnate -> switch"
                )
            keep = state.current_af if state.current_af in pool else pool[0]
            return emit(keep, "early_keep", "early: keep current")

        # --- MID: LogEI is the workhorse; exploration is a committed burst ----
        # If we are mid-burst, stay on the committed explorer (patience is ignored
        # for it -- an explorer is *supposed* to pick away from the incumbent, so
        # we don't punish it for that). Bail early if it actually improved.
        burst = self.history.explore_burst_remaining
        if burst > 0:
            self.history.explore_burst_remaining = burst - 1
            if not record.is_improving:
                return emit(state.current_af, "mid_explore", "committed explore burst")
            self.history.explore_burst_remaining = 0
            self._retire_af(state.current_af)
            return emit("LogEI", "mid_exploit", "burst found improvement -> LogEI")

        # Not in a burst: LogEI exploits by default. We only leave it to explore
        # when the incumbent has PLATEAUED -- no improvement for `stall_patience`
        # exploit iterations -- and there is still uncertainty worth probing.
        # (Per-iteration improvement is too rare in BO to drive the decision.)
        stalled = self.history.iters_since_improvement >= cfg.stall_patience
        if stalled and record.current_uncertainty >= record.uncertainty_lower:
            # one explorer for the whole burst: aggressive UCB if very uncertain,
            # else information-gathering MES.
            explorer = (
                "UCB"
                if record.current_uncertainty >= record.uncertainty_upper
                else "MES"
            )
            self.history.explore_burst_remaining = cfg.explore_burst - 1
            self.history.iters_since_improvement = 0  # LogEI gets a fresh runway after
            self._retire_af(
                state.current_af
            )  # leaving LogEI: clean slate for its return
            return emit(explorer, "mid_explore_start", f"plateau -> explore {explorer}")

        # Otherwise exploit with LogEI (switch onto it if we drifted off it).
        if state.current_af != "LogEI":
            self._retire_af(state.current_af)
        return emit("LogEI", "mid_exploit", "LogEI workhorse")


if __name__ == "__main__":
    # Archive the verbatim defaults so rule_versions/v0_verbatim.json always exists.
    print("saved verbatim config ->", save_config(HeuristicConfig()))
