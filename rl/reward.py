"""
Reward computation for the NoC RL training loop.

Baseline latencies measured on: examples/mesh88_lat, routing_function=dor,
traffic=<pattern>, seed=1, sim_count=3. All converged cleanly ((3 samples)
tag). See PROJECT_HANDOFF.md for the sessions that produced these numbers
and the reasoning behind the reward formula.

UPDATE (this session): each traffic pattern has its own stable injection-rate
range and its own baseline table -- discovered when hotspot(0) showed 0%
convergence at uniform's rates, which turned out to be ~25x too high for
hotspot's actual saturation point. Checked all patterns the same way;
transpose, bitcomp, and tornado were also narrower than uniform, just less
dramatically. See the measured ranges below.
"""

# (injection_rate, dor+Active baseline latency) per traffic pattern.
# Each table only covers its own pattern's validated range -- do not use
# one pattern's table to score another pattern's episodes.
BASELINE_TABLES = {
    "uniform": [
        (0.001, 47.32),
        (0.005, 54.71),
        (0.010, 70.39),
        (0.015, 103.10),
        (0.020, 232.06),
    ],
    "tornado": [
        (0.001, 56.958),
        (0.005, 69.4002),
        (0.010, 105.583),
        (0.011, 116.139),
        (0.012, 141.563),
        (0.013, 171.43),
        (0.014, 243.028),
    ],
    "bitcomp": [
        (0.001, 59.1702),
        (0.005, 75.1288),
        (0.010, 162.66),
        (0.011, 214.11),
    ],
    "transpose": [
        (0.001, 47.414),
        (0.005, 63.3402),
        (0.006, 80.2354),
        (0.007, 104.262),
    ],
    "hotspot(0)": [
        (0.0002, 59.4573),
        (0.0004, 72.4292),
        (0.0006, 107.293),
        (0.0008, 373.569),
    ],
}

# Per-pattern valid injection-rate range, used for sampling during training
# and for picking eval rates. Do not sample or evaluate outside these
# ranges -- the pattern reliably fails to converge past its upper bound
# (measured this session).
PATTERN_RATE_RANGES = {
    "uniform":    (0.001, 0.020),
    "tornado":    (0.001, 0.014),
    "bitcomp":    (0.001, 0.011),
    "transpose":  (0.001, 0.007),
    "hotspot(0)": (0.0002, 0.0008),
}

TRAFFIC_PATTERNS = ["uniform", "transpose", "bitcomp", "tornado", "hotspot(0)"]


def sample_traffic_pattern(rng):
    return rng.choice(TRAFFIC_PATTERNS)


# Orion3-measured power per mode, mW. Mode indices match decode_action()'s
# mode = action // 4 encoding (0=Active, 1=Balanced, 2=Eco).
POWER_BY_MODE = {0: 226.99, 1: 152.63, 2: 111.09}
BASELINE_POWER = POWER_BY_MODE[0]  # Active = "no savings" reference

ENERGY_WEIGHT = 0.4  # tunable; sweeping this is a natural experiment for the report


def baseline_latency_for(injection_rate: float, pattern: str) -> float:
    """Linearly interpolate the dor+Active baseline latency for a given
    injection rate, within the given pattern's own measured table. Clamps
    to the table's edges outside its measured range -- do not train above
    a pattern's PATTERN_RATE_RANGES upper bound until that pattern's table
    is extended with new measurements."""
    table = BASELINE_TABLES[pattern]
    if injection_rate <= table[0][0]:
        return table[0][1]
    if injection_rate >= table[-1][0]:
        return table[-1][1]
    for (r0, l0), (r1, l1) in zip(table, table[1:]):
        if r0 <= injection_rate <= r1:
            frac = (injection_rate - r0) / (r1 - r0)
            return l0 + frac * (l1 - l0)


def sample_injection_rate_for_pattern(rng, pattern: str) -> float:
    """Sample one episode's injection rate for the given traffic pattern,
    weighted toward the upper half of that pattern's validated range so the
    agent sees the load-dependent trade-off regime often enough, not just
    the light-load regime where mode choice barely matters. rng is a numpy
    Generator or random.Random with .random().

    Weighting: square-root the uniform sample before scaling, which biases
    toward the high end of the pattern's range while still covering the
    low end. Same weighting scheme as the original uniform-only sampler,
    now applied per-pattern since each pattern's range differs wildly
    (hotspot's range is ~25x narrower than uniform's).
    """
    lo, hi = PATTERN_RATE_RANGES[pattern]
    u = rng.random()
    biased = u ** 0.5
    return lo + biased * (hi - lo)


def compute_reward(episode_latency: float, injection_rate: float,
                    mode_choices, pattern: str) -> float:
    """
    episode_latency: the final 'Packet latency average' value BookSim2
        reported for this episode (parsed from stdout).
    injection_rate: the rate this episode was run at.
    mode_choices: list/iterable of mode ints (0/1/2) chosen across every
        decision served during the episode -- i.e. action // 4 for each
        buffered action (full RL), or the action itself (power-only agents).
    pattern: the traffic pattern this episode ran, e.g. "uniform",
        "hotspot(0)" -- selects which baseline table to score against.

    Returns a single float reward for the whole episode. More negative is
    worse; closer to 0 is better. There is no explicit wake-up-penalty term
    -- that cost is already real, physically simulated latency (validated
    earlier this project), so it's folded into episode_latency already.
    """
    baseline_lat = baseline_latency_for(injection_rate, pattern)
    mode_list = list(mode_choices)
    if not mode_list:
        # No decisions were served (shouldn't normally happen) -- fall back
        # to a pure latency penalty with no energy term.
        return -(episode_latency / baseline_lat)
    avg_power = sum(POWER_BY_MODE[m] for m in mode_list) / len(mode_list)
    return -(episode_latency / baseline_lat) - ENERGY_WEIGHT * (avg_power / BASELINE_POWER)


# Episodes that never converge (BookSim2 aborts via latency_thres, no
# "(N samples)" line ever printed) need a defined fallback -- NOT YET
# DECIDED, see handoff doc §5 open items. Placeholder for now:
NONCONVERGENCE_PENALTY = -10.0  # TODO: revisit -- is this harsh enough / too harsh?
