"""
Reward computation for the NoC RL training loop.

Baseline latencies measured on: examples/mesh88_lat, routing_function=dor,
traffic=uniform, seed=1, sim_count=3. All converged cleanly ((3 samples) 
tag).
See PROJECT_HANDOFF.md for the session that produced these numbers and the
reasoning behind the reward formula.
"""

# (injection_rate, dor+Active baseline latency)
BASELINE_LATENCY = [
    (0.001, 47.32),
    (0.005, 54.71),
    (0.010, 70.39),
    (0.015, 103.10),
    (0.020, 232.06),
]

# Orion3-measured power per mode, mW. Mode indices match decode_action()'s
# mode = action // 4 encoding (0=Active, 1=Balanced, 2=Eco).
POWER_BY_MODE = {0: 226.99, 1: 152.63, 2: 111.09}
BASELINE_POWER = POWER_BY_MODE[0]  # Active = "no savings" reference

ENERGY_WEIGHT = 0.4  # tunable; sweeping this is a natural experiment for the report


def baseline_latency_for(injection_rate: float) -> float:
    """Linearly interpolate the dor+Active baseline latency for a given
    injection rate. Clamps to the table's edges outside its measured range
    ([0.001, 0.020] as of this session -- do not train above 0.020 until
    this table is extended; see handoff doc for why)."""
    table = BASELINE_LATENCY
    if injection_rate <= table[0][0]:
        return table[0][1]
    if injection_rate >= table[-1][0]:
        return table[-1][1]
    for (r0, l0), (r1, l1) in zip(table, table[1:]):
        if r0 <= injection_rate <= r1:
            frac = (injection_rate - r0) / (r1 - r0)
            return l0 + frac * (l1 - l0)


def sample_injection_rate(rng) -> float:
    """Sample one episode's injection rate, weighted toward the upper half
    of the validated range so the agent sees the load-dependent trade-off
    regime often enough, not just the light-load regime where mode choice
    barely matters. rng is a numpy Generator or random.Random with 
.random().

    Weighting: square the uniform sample before scaling, which biases
    toward the high end of [0.001, 0.020] while still covering the low 
end.
    """
    lo, hi = BASELINE_LATENCY[0][0], BASELINE_LATENCY[-1][0]
    u = rng.random()
    biased = u ** 0.5  # pulls samples up toward hi; tune exponent if needed
    return lo + biased * (hi - lo)


def compute_reward(episode_latency: float, injection_rate: float, 
mode_choices) -> float:
    """
    episode_latency: the final 'Packet latency average' value BookSim2
        reported for this episode (parsed from stdout).
    injection_rate: the rate this episode was run at.
    mode_choices: list/iterable of mode ints (0/1/2) chosen across every
        decision served during the episode -- i.e. action // 4 for each
        buffered action.

    Returns a single float reward for the whole episode. More negative is
    worse; closer to 0 is better. There is no explicit wake-up-penalty 
term
    -- that cost is already real, physically simulated latency (validated
    this session), so it's folded into episode_latency already.
    """
    baseline_lat = baseline_latency_for(injection_rate)
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
