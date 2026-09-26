"""Exact robust (interval-MDP) maximum-reachability solver over rationals.

Each controllable action gives, per possible successor, a *closed interval*
``[lower, upper]`` of legal transition probabilities.  After every action the
adversary (the "link disturbance") may afresh choose any distribution over the
successors that (a) respects every interval and (b) sums to exactly one.  The
controller chooses actions to *maximise* the infinite-horizon rescue
probability while the adversary *minimises* it and may re-choose after every
single action.

Game model and exact solution (no value iteration, no midpoints, no floats)
---------------------------------------------------------------------------

Expanding every action into an adversarial decision node turns the model into
a finite *turn-based stochastic reachability game*: at a state the controller
selects an action; at the action node the adversary selects a distribution
from the interval box.  For such games both players have stationary
deterministic optimal strategies, where "deterministic" for the adversary
means picking a *vertex* of its feasible box (a polytope whose vertices put
every successor at ``lower`` or ``upper`` except at most one).

The robust Bellman operator is

    (T v)(s) = max_a  min_{x in I(s,a)} sum_t x_t v(t),

with ``v(r) = 1`` on rescued terminals; the game value is its least fixed
point.  This module computes that fixed point exactly by *adversary strategy
iteration*:

  1. fix one vertex distribution ``tau(s, a)`` per action node,
  2. the controller then faces an ordinary MDP, whose value vector is the
     least fixed point of  v(s) = max_a sum_t tau(s,a,t) v(t)  -- computed
     *exactly* by the rational LP in :mod:`app.solver` (two-phase simplex in
     ``fractions.Fraction``),
  3. best-respond for the adversary: at every action node replace ``tau`` by
     the box distribution minimising the expectation against the current
     exact value vector,
  4. repeat until no action node can be improved.

The MDP value vectors form a componentwise non-increasing sequence of exact
rationals bounded below by the game value; every strict step moves to a
different adversary policy among the finitely many vertex policies, so the
loop terminates after finitely many iterations -- never an approximate limit.
At the fixed point ``tau(s, a)`` attains the box minimum for every action and
``v(s) = (T v)(s)``, i.e. the exact robust value, with the controller's
canonical optimal actions read off directly.

Canonical worst distribution
----------------------------

For a fixed value vector the adversary's problem is the box-linear program

    minimise  sum_t x_t v(t)
    such that sum_t x_t = 1,  lower_t <= x_t <= upper_t,

whose exact optimum is obtained by water-filling: reserve the mandatory mass
``lower_t`` everywhere, then grant the remaining pool
``1 - sum lower`` to successors in ascending order of ``(value, identifier)``,
saturating each cap ``upper_t - lower_t``.  The identifier tie-break makes the
minimising vertex unique and canonical even when several successors share a
value, which is what makes strategy iteration deterministic.
"""

from fractions import Fraction

from .solver import non_terminating_states, solve_reachability


def interval_distribution_feasible(intervals):
    """Return whether ``{target: (lower, upper)}`` admits a distribution.

    The polytope ``lower_t <= x_t <= upper_t`` with ``sum x_t = 1`` is
    non-empty exactly when ``lower <= upper`` coordinatewise and
    ``sum lower <= 1 <= sum upper`` (the mandatory mass and the total capacity
    are the only coupling constraints).
    """
    if not intervals:
        return False
    lower_sum = Fraction(0)
    upper_sum = Fraction(0)
    for lower, upper in intervals.values():
        if lower > upper:
            return False
        lower_sum += lower
        upper_sum += upper
    return lower_sum <= 1 <= upper_sum


def worst_distribution(targets, lower, upper, values):
    """Canonical distribution minimising ``sum x_t values[t]``.

    ``targets``       iterable of target identifiers
    ``lower/upper``   ``{target: Fraction}`` interval endpoints
    ``values``        ``{state: Fraction}`` value vector to evaluate against

    Returns ``(distribution, expected_value)`` where ``distribution`` maps
    every target to its chosen :class:`Fraction` probability, lies inside the
    intervals, sums to exactly one and attains the minimum expectation.  The
    free probability pool is granted in ascending ``(value, identifier)``
    order (water-filling), so the result is the canonical minimising vertex.
    """
    chosen = {t: lower[t] for t in targets}
    pool = Fraction(1) - sum(lower.values(), Fraction(0))
    for t in sorted(targets, key=lambda t: (values[t], t)):
        if pool == 0:
            break
        cap = upper[t] - lower[t]
        take = pool if pool < cap else cap
        if take:
            chosen[t] += take
            pool -= take
    expected = sum((chosen[t] * values[t] for t in targets), Fraction(0))
    return chosen, expected


def solve_robust_reachability(states, start, rescued, lost, actions):
    """Solve one interval model exactly by adversary strategy iteration.

    ``actions`` maps ``{state: {action: {target: (lower, upper)}}}`` for
    controllable states only; every interval family is assumed feasible
    (validated by the caller).

    Returns a dict with the same outer shape as
    :func:`app.solver.solve_reachability`; each certificate action carries the
    exact endpoints, the canonical worst-case distribution and its expectation:
      ``maxRescueProbability``  robust value of the start state
      ``stateValues``           ``{state: Fraction}`` for every state
      ``optimalActions``        canonical maximising action per state
      ``certificates``          per-state/per-action itemised certificates
      ``nonTerminatingStates``  states unable to reach any terminal even under
                                the union of feasible interval supports
    """
    rescued_set = set(rescued)
    lost_set = set(lost)
    terminals = rescued_set | lost_set

    # Precompute per-action target order and endpoint views.
    boxes = {}
    for s, state_actions in actions.items():
        boxes[s] = {}
        for action_id, intervals in state_actions.items():
            targets = sorted(intervals)
            boxes[s][action_id] = (
                targets,
                {t: intervals[t][0] for t in targets},
                {t: intervals[t][1] for t in targets},
            )

    # Initial adversary policy: the canonical minimising vertex against the
    # pessimistic value vector (rescued == 1, everything else 0).
    baseline = {s: Fraction(1 if s in rescued_set else 0) for s in states}
    tau = {}
    for s, state_actions in actions.items():
        tau[s] = {}
        for action_id, (targets, lower, upper) in boxes[s].items():
            distribution, _ = worst_distribution(
                targets, lower, upper, baseline)
            tau[s][action_id] = distribution

    # Strategy iteration: solve the controller's MDP under the fixed adversary
    # policy exactly, then let the adversary best-respond at every node.
    seen_policies = set()
    while True:
        result = solve_reachability(states, start, rescued, lost, tau)
        values = result["stateValues"]

        improved = False
        for s, state_actions in actions.items():
            for action_id, (targets, lower, upper) in boxes[s].items():
                current = tau[s][action_id]
                current_value = sum((current[t] * values[t]
                                     for t in targets), Fraction(0))
                distribution, worst_value = worst_distribution(
                    targets, lower, upper, values)
                if worst_value < current_value:
                    tau[s][action_id] = distribution
                    improved = True
        if not improved:
            break

        # Determinism guard: the all-switch improvement rule is strictly
        # monotone, so a recurring policy would indicate a bug, not input.
        signature = tuple(
            (s, a, tuple(sorted(tau[s][a].items())))
            for s in sorted(tau) for a in sorted(tau[s]))
        if signature in seen_policies:
            raise RuntimeError("robust strategy iteration cycled")
        seen_policies.add(signature)

    values = result["stateValues"]

    # Controller certificate: each action's value is its worst-case expected
    # value under the final exact vector.
    certificates = []
    optimal_actions = {}
    for s in states:
        if s in terminals:
            continue
        entries = []
        for action_id in sorted(actions[s]):
            targets, lower, upper = boxes[s][action_id]
            distribution, expected = worst_distribution(
                targets, lower, upper, values)
            entries.append({
                "action": action_id,
                "expectedValue": expected,
                "optimal": expected == values[s],
                "targets": targets,
                "lower": lower,
                "upper": upper,
                "worstDistribution": distribution,
            })
        chosen = min(e["action"] for e in entries if e["optimal"])
        for e in entries:
            e["selected"] = e["action"] == chosen
        optimal_actions[s] = chosen
        certificates.append({
            "state": s,
            "value": values[s],
            "selectedAction": chosen,
            "actions": entries,
        })

    # States unable to reach any terminal on the union of feasible supports.
    # A target admits strictly positive mass in *some* feasible distribution
    # iff upper_t > 0 and the other mandatory lower masses do not already fill
    # the whole unit (sum lower - lower_t < 1); when sum lower = 1 every
    # zero-lower target is forced to zero no matter how large its cap is.
    support_actions = {}
    for s, state_actions in actions.items():
        edge_actions = {}
        for action_id, intervals in state_actions.items():
            lower_sum = sum((lo for lo, _hi in intervals.values()),
                            Fraction(0))
            edge_actions[action_id] = {
                t: Fraction(1)
                for t, (lo, hi) in intervals.items()
                if hi > 0 and lower_sum - lo < 1}
        support_actions[s] = edge_actions

    return {
        "start": start,
        "maxRescueProbability": values[start],
        "stateValues": values,
        "optimalActions": optimal_actions,
        "certificates": certificates,
        "nonTerminatingStates": non_terminating_states(
            states, terminals, support_actions),
    }
