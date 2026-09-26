"""Exact maximum-reachability solver for finite controllable models (MDPs)
over rational transition probabilities.

The maximal rescue (reachability) probability vector of a finite MDP is the
least fixed point of the Bellman optimality operator

    (T v)(s) = max_a  sum_{s'} p(s, a, s') * v(s')

with v(rescued) = 1 and v >= 0.  That least fixed point is the unique
componentwise-minimal point of the feasible polytope

    v(s) >= sum_{s'} p(s, a, s') * v(s')   for every controllable s, action a
    v(r) >= 1                              for every rescued terminal r
    v(s) >= 0                              for every state s

so it is obtained *exactly* by solving the linear program

    minimise  sum_s v(s)   subject to the constraints above.

This module solves that LP with a two-phase simplex method carried out
entirely in ``fractions.Fraction`` arithmetic (Bland's pivoting rule, so
termination is guaranteed).  No floating point, no random simulation, no
enumeration of policies and no truncated value iteration is involved: the
simplex method is an exact algebraic algorithm that ends after finitely many
rational pivots with the provably optimal solution.

States that can never reach any terminal (rescued or lost) -- e.g. closed
loops with no exit -- are additionally identified by an exact graph
reachability pass over the support of the transition distributions.

Robust (interval-valued) models
-------------------------------
``solve_robust_reachability`` generalises the audit to interval MDPs: every
transition carries a reduced-fraction lower/upper bound, and after every
action the perturbation may pick ANY distribution inside the intervals --
adversarially, and re-chosen on every visit.  The guaranteed (robust) rescue
probability is the value of the controller-maximising / perturbation-
minimising infinite-horizon reachability game, i.e. the least fixed point of

    (T v)(s) = max_a  min_{p in Delta_a}  sum_{s'} p(s') * v(s')

with v(rescued) = 1 and v >= 0, where Delta_a is the interval polytope
``{p : lo <= p <= hi, sum p = 1}``.  Unlike the plain max operator, the
pre-fixed points of this max-min operator do NOT form a polytope (the inner
minimum makes them non-convex), so a single LP cannot express them.  The
value is instead computed by exact controller strategy iteration:

* given a controller strategy, the remaining min-reachability MDP is solved
  exactly: the adversary's safe region (states where it can force the chain
  to never reach a rescued terminal) is an exact graph fixpoint with value
  0; outside it every adversary strategy is proper, the Bellman fixed point
  is unique, and it equals the greatest sub-fixed point of
  ``v(s) <= min_p p . v`` -- obtained by MAXIMISING ``sum_s v(s)`` over the
  vertex constraints ``v(s) <= p . v`` with the same exact rational simplex,
  the (too many) interval vertices being added lazily by constraint
  generation: each round's worst vertex is found exactly by a greedy
  minimisation (every target at its lower bound, the remaining mass poured
  onto the lowest-value targets);
* the controller then switches every state with a strictly better
  worst-case expectation to its canonical best action.  Values increase
  componentwise monotonically, so finitely many switches end at a greedy
  strategy whose value is exactly the game value.

No interval midpoints, no floating point, no random replay and no truncated
finite-round iteration are involved anywhere.
"""

from fractions import Fraction


class SolverError(Exception):
    """Raised on internal solver inconsistencies (infeasible/unbounded LP).

    The reachability LP is always feasible (v == 1 everywhere satisfies every
    constraint) and bounded below by 0, so this signals a bug, not bad input.
    """


def _simplex_min(c, A, b):
    """Solve ``min c.x`` subject to ``A x >= b`` and ``x >= 0`` exactly.

    ``c``, ``A`` and ``b`` must contain :class:`fractions.Fraction` values.
    Returns the optimal solution vector as a list of Fractions.
    """
    n = len(c)
    m = len(A)
    if m == 0:
        return [Fraction(0)] * n

    A = [[Fraction(x) for x in row] for row in A]
    b = [Fraction(x) for x in b]
    # Normalise to non-negative right-hand sides.  A row  A x >= b  with
    # b < 0 is multiplied by -1, which flips it into a '<=' row; the slack
    # variable of such a row enters with coefficient +1 (A x + s = b),
    # whereas '>=' rows keep coefficient -1 (A x - s = b).
    slack_sign = []
    for i in range(m):
        if b[i] < 0:
            A[i] = [-x for x in A[i]]
            b[i] = -b[i]
            slack_sign.append(Fraction(1))
        else:
            slack_sign.append(Fraction(-1))

    # Equality form:  A x +/- slack = b,  plus one artificial variable per
    # row to bootstrap phase 1.  Column layout:
    #   [0, n)             structural variables x
    #   [n, n + m)         slack variables
    #   [n + m, n + 2m)    artificial variables
    width = n + 2 * m
    rhs = width  # index of the right-hand-side column
    tableau = []
    for i in range(m):
        row = [Fraction(0)] * (width + 1)
        for j in range(n):
            row[j] = A[i][j]
        row[n + i] = slack_sign[i]
        row[n + m + i] = Fraction(1)
        row[rhs] = b[i]
        tableau.append(row)
    basis = [n + m + i for i in range(m)]
    artificial_from = n + m

    def pivot(row_ix, col):
        prow = tableau[row_ix]
        inv = Fraction(1) / prow[col]
        tableau[row_ix] = [x * inv for x in prow]
        for i in range(len(tableau)):
            if i != row_ix:
                factor = tableau[i][col]
                if factor:
                    tableau[i] = [x - factor * y
                                  for x, y in zip(tableau[i], tableau[row_ix])]
        basis[row_ix] = col

    def optimize(costs, allowed):
        # Bland's rule on both entering and leaving variables: terminates.
        while True:
            rows = len(tableau)
            cost_b = [costs[basis[i]] for i in range(rows)]
            enter = -1
            for j in range(width):
                if j not in allowed:
                    continue
                # reduced cost r_j = c_B . T[:, j] - c_j ; r_j > 0 improves
                acc = -costs[j]
                for i in range(rows):
                    if cost_b[i]:
                        acc += cost_b[i] * tableau[i][j]
                if acc > 0:
                    enter = j
                    break
            if enter < 0:
                return
            leave = -1
            best_ratio = None
            for i in range(rows):
                coeff = tableau[i][enter]
                if coeff > 0:
                    ratio = tableau[i][rhs] / coeff
                    if (best_ratio is None or ratio < best_ratio
                            or (ratio == best_ratio and basis[i] < basis[leave])):
                        best_ratio = ratio
                        leave = i
            if leave < 0:
                raise SolverError("reachability LP is unbounded")
            pivot(leave, enter)

    # ---- Phase 1: minimise the sum of artificial variables ----
    cost1 = [Fraction(0)] * width
    for j in range(artificial_from, width):
        cost1[j] = Fraction(1)
    optimize(cost1, set(range(width)))
    phase1_obj = sum(cost1[basis[i]] * tableau[i][rhs]
                     for i in range(len(tableau)))
    if phase1_obj != 0:
        raise SolverError("reachability LP is infeasible")

    # Drive artificial variables out of the basis; drop redundant rows.
    i = 0
    while i < len(tableau):
        if basis[i] >= artificial_from:
            col = -1
            for j in range(artificial_from):
                if tableau[i][j] != 0:
                    col = j
                    break
            if col >= 0:
                pivot(i, col)
                i += 1
            else:
                del tableau[i]
                del basis[i]
        else:
            i += 1

    # ---- Phase 2: minimise the true objective; artificials never re-enter ----
    cost2 = [Fraction(0)] * width
    for j in range(n):
        cost2[j] = Fraction(c[j])
    optimize(cost2, set(range(artificial_from)))

    x = [Fraction(0)] * n
    for i in range(len(tableau)):
        if basis[i] < n:
            x[basis[i]] = tableau[i][rhs]
    return x


def _non_terminating_from_support(states, terminals, successors):
    """States from which no terminal is reachable in the support graph."""
    can_reach = set(terminals)
    changed = True
    while changed:
        changed = False
        for s in states:
            if s not in can_reach and successors.get(s, set()) & can_reach:
                can_reach.add(s)
                changed = True
    return sorted(s for s in states if s not in can_reach)


def non_terminating_states(states, terminals, actions):
    """Return the sorted list of states that can never reach ANY terminal.

    A state can reach a terminal if a path exists in the support graph, whose
    edges are every transition with a strictly positive probability under some
    action.  States that cannot are trapped forever in non-terminal closed
    loops (they are rescued with probability 0 and lost with probability 0).
    """
    successors = {s: set() for s in states}
    for state, state_actions in actions.items():
        for dist in state_actions.values():
            for target, prob in dist.items():
                if prob > 0:
                    successors[state].add(target)
    return _non_terminating_from_support(states, terminals, successors)


def non_terminating_states_interval(states, terminals, actions):
    """Interval-model analogue of :func:`non_terminating_states`.

    ``actions`` maps ``state -> action -> {target: (lower, upper)}``.  A
    transition is in the support iff SOME distribution inside the action's
    interval polytope gives it strictly positive probability, i.e. iff
    ``min(upper_t, 1 - sum(lo) + lo_t) > 0`` (all other targets pinned to
    their lower bounds, every remaining unit of mass poured onto ``t``).
    For point intervals this coincides with ``prob > 0``.
    """
    successors = {s: set() for s in states}
    for state, state_actions in actions.items():
        for bounds in state_actions.values():
            lower_sum = sum((lo for lo, _hi in bounds.values()), Fraction(0))
            for target, (lo, hi) in bounds.items():
                if min(hi, Fraction(1) - lower_sum + lo) > 0:
                    successors[state].add(target)
    return _non_terminating_from_support(states, terminals, successors)


def solve_reachability(states, start, rescued, lost, actions):
    """Solve one audit model exactly.

    ``states``   ordered list of unique state identifiers (validated already)
    ``start``    the starting state identifier
    ``rescued``  collection of rescued terminal identifiers (value 1)
    ``lost``     collection of lost terminal identifiers (value 0)
    ``actions``  ``{state: {action: {target: Fraction}}}`` for controllable
                 states only

    Returns a dict of exact results (Fractions, not strings):
      ``maxRescueProbability``  value of the start state
      ``stateValues``           {state: Fraction} for every state
      ``optimalActions``        {state: canonical action} per controllable state
      ``certificates``          per-state itemised certificates
      ``nonTerminatingStates``  sorted list of states that never reach a terminal
    """
    rescued_set = set(rescued)
    lost_set = set(lost)
    terminals = rescued_set | lost_set
    index = {s: i for i, s in enumerate(states)}
    n = len(states)

    rows = []
    rhs = []
    for s in states:
        if s in terminals:
            continue
        for action_id in sorted(actions[s]):
            dist = actions[s][action_id]
            row = [Fraction(0)] * n
            row[index[s]] = Fraction(1)
            for target, prob in dist.items():
                row[index[target]] -= prob
            rows.append(row)
            rhs.append(Fraction(0))
    for r in rescued:
        row = [Fraction(0)] * n
        row[index[r]] = Fraction(1)
        rows.append(row)
        rhs.append(Fraction(1))

    objective = [Fraction(1)] * n
    solution = _simplex_min(objective, rows, rhs)
    values = {s: solution[index[s]] for s in states}

    certificates = []
    optimal_actions = {}
    for s in states:
        if s in terminals:
            continue
        entries = []
        for action_id in sorted(actions[s]):
            dist = actions[s][action_id]
            expected = sum((prob * values[target]
                            for target, prob in dist.items()), Fraction(0))
            entries.append({
                "action": action_id,
                "expectedValue": expected,
                "optimal": expected == values[s],
            })
        # Canonical choice: among ALL optimal actions, the smallest identifier.
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

    return {
        "start": start,
        "maxRescueProbability": values[start],
        "stateValues": values,
        "optimalActions": optimal_actions,
        "certificates": certificates,
        "nonTerminatingStates": non_terminating_states(states, terminals, actions),
    }


def _canonical_worst_distribution(bounds, values):
    """Minimise ``sum_t p(t) * values[t]`` over one action's interval polytope.

    ``bounds``  ``{target: (lower, upper)}`` with ``sum(lower) <= 1 <=
    sum(upper)`` (validated before the solver runs)
    ``values``  ``{state: Fraction}`` current value vector

    Returns ``(minimum, minimiser)`` where the minimiser is the canonical
    worst vertex: every target starts at its lower bound and the remaining
    mass ``1 - sum(lower)`` is poured onto the lowest-value targets first,
    each up to its upper bound.  Ties in value are broken by target
    identifier, so the vertex is a deterministic, canonical choice among all
    minimisers.  All arithmetic is exact rational arithmetic.
    """
    lower_sum = sum((lo for lo, _hi in bounds.values()), Fraction(0))
    remaining = Fraction(1) - lower_sum
    if remaining < 0:
        raise SolverError("interval lower bounds sum to more than 1")
    dist = {t: lo for t, (lo, _hi) in bounds.items()}
    for t in sorted(bounds, key=lambda t: (values[t], t)):
        lo, hi = bounds[t]
        pour = hi - lo
        if pour > remaining:
            pour = remaining
        if pour:
            dist[t] += pour
            remaining -= pour
    if remaining:
        raise SolverError("interval upper bounds sum to less than 1")
    expected = sum((dist[t] * values[t] for t in bounds), Fraction(0))
    return expected, dist


def _adversary_safe_region(states, rescued, sigma_actions):
    """States from which the adversary can force NEVER reaching ``rescued``.

    ``sigma_actions`` maps every controllable state to the single interval
    distribution ``{target: (lower, upper)}`` selected by the controller.
    The safe region is the greatest fixed point of

        Safe(X) = {s not rescued : s has no actions (a lost terminal), or
                   some p in Delta_s has supp(p) subset of X}

    computed exactly as a graph fixpoint.  A legal distribution supported
    inside X exists iff every target outside X has lower bound 0 and the
    upper bounds inside X sum to at least 1.  On these states the
    perturbation can keep the chain inside X (or absorb it in a lost
    terminal) forever, so the robust rescue probability is exactly 0.
    """
    safe = set(states) - set(rescued)
    changed = True
    while changed:
        changed = False
        for s in sorted(safe):
            if s not in sigma_actions:
                continue  # lost terminal: never reaches 'rescued'
            bounds = sigma_actions[s]
            forced_outside = any(lo > 0 for t, (lo, _hi) in bounds.items()
                                 if t not in safe)
            hi_inside = sum((hi for t, (_lo, hi) in bounds.items()
                             if t in safe), Fraction(0))
            if forced_outside or hi_inside < 1:
                safe.discard(s)
                changed = True
    return safe


def _min_mdp_value(states, rescued, sigma_actions):
    """Exact perturbation-minimal rescue probability under a FIXED controller
    strategy (a min-reachability MDP over interval distributions).

    ``sigma_actions``  ``{state: {target: (lower, upper)}}`` for every
    controllable state.  Returns ``{state: Fraction}`` for all states.

    On the adversary's safe region the value is 0 (exact graph fixpoint).
    Everywhere else every adversary vertex strategy reaches a rescued
    terminal almost surely (a closed class avoiding it would itself be a
    safe region), so the Bellman operator  v = min_p p . v  has a UNIQUE
    fixed point there.  That point is the greatest sub-fixed point of
    ``v(s) <= min_p p . v``, obtained exactly by MAXIMISING ``sum_s v(s)``
    over the vertex constraints  ``v(s) <= p . v``  -- added lazily by
    constraint generation, since each interval polytope has too many
    vertices to enumerate.  All arithmetic is exact rational arithmetic.
    """
    rescued_set = set(rescued)
    safe = _adversary_safe_region(states, rescued_set, sigma_actions)
    values = {s: (Fraction(1) if s in rescued_set else Fraction(0))
              for s in states}

    # LP variables: controllable states outside the safe region.  (Lost
    # terminals are always safe, and rescued terminals are constant 1.)
    free = [s for s in states
            if s not in safe and s not in rescued_set]
    if not free:
        return values
    index = {s: i for i, s in enumerate(free)}
    n = len(free)

    rows = []
    rhs = []
    for s in free:  # v(s) <= 1
        row = [Fraction(0)] * n
        row[index[s]] = Fraction(-1)
        rows.append(row)
        rhs.append(Fraction(-1))

    # Maximise sum_s v(s)  <=>  minimise -sum_s v(s).
    objective = [Fraction(-1)] * n
    while True:
        solution = _simplex_min(objective, rows, rhs)
        for s, i in index.items():
            values[s] = solution[i]
        violated = False
        for s in free:
            expected, vertex = _canonical_worst_distribution(
                sigma_actions[s], values)
            if values[s] > expected:
                # add  v(s) <= sum_t p(t) v(t) + const  with the constants
                # being the rescued terminals' share
                row = [Fraction(0)] * n
                row[index[s]] = Fraction(-1)
                const = Fraction(0)
                for target, prob in vertex.items():
                    if target in index:
                        row[index[target]] += prob
                    elif target in rescued_set:
                        const += prob
                    # targets in the safe region contribute value 0
                rows.append(row)
                rhs.append(-const)
                violated = True
        if not violated:
            return values


def solve_robust_reachability(states, start, rescued, lost, actions):
    """Solve one robust (interval-valued) audit model exactly.

    ``states``   ordered list of unique state identifiers (validated already)
    ``start``    the starting state identifier
    ``rescued``  collection of rescued terminal identifiers (value 1)
    ``lost``     collection of lost terminal identifiers (value 0)
    ``actions``  ``{state: {action: {target: (lower, upper)}}}`` for
                 controllable states only; every action's intervals admit a
                 distribution summing to exactly one (validated already)

    The result is the controller-maximising / perturbation-minimising
    infinite-horizon rescue probability: after every action the perturbation
    may re-select any distribution inside that action's intervals.  It is
    computed by controller strategy iteration: from a controller strategy,
    evaluate the resulting min-reachability MDP EXACTLY
    (:func:`_min_mdp_value`), then switch every state whose best worst-case
    expectation strictly improves to its canonical best action.  Values
    increase componentwise monotonically, so finitely many switches reach a
    greedy strategy, whose value is a fixed point of the max-min Bellman
    operator and bounded below by every fixed point of the operator
    restricted to that strategy -- i.e. exactly the game value.  Returns the
    same dict shape as :func:`solve_reachability`, with one extra
    ``worstDistribution`` entry per certificate action: the canonical
    worst-case distribution attaining that action's expected value.
    """
    terminals = set(rescued) | set(lost)
    controllable = [s for s in states if s not in terminals]

    sigma = {s: sorted(actions[s])[0] for s in controllable}
    seen_strategies = set()
    while True:
        key = tuple(sorted(sigma.items()))
        if key in seen_strategies:
            raise SolverError("controller strategy iteration cycled")
        seen_strategies.add(key)
        sigma_actions = {s: actions[s][sigma[s]] for s in controllable}
        values = _min_mdp_value(states, rescued, sigma_actions)
        improved = False
        new_sigma = dict(sigma)
        for s in controllable:
            best_action = None
            best_expected = None
            for action_id in sorted(actions[s]):
                expected, _vertex = _canonical_worst_distribution(
                    actions[s][action_id], values)
                if best_expected is None or expected > best_expected:
                    best_action = action_id
                    best_expected = expected
            if best_expected > values[s]:
                new_sigma[s] = best_action
                improved = True
        if not improved:
            break
        sigma = new_sigma

    certificates = []
    optimal_actions = {}
    for s in states:
        if s in terminals:
            continue
        entries = []
        for action_id in sorted(actions[s]):
            expected, vertex = _canonical_worst_distribution(
                actions[s][action_id], values)
            entries.append({
                "action": action_id,
                "expectedValue": expected,
                "worstDistribution": vertex,
                "optimal": expected == values[s],
            })
        # Canonical choice: among ALL optimal actions, the smallest identifier.
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

    return {
        "start": start,
        "maxRescueProbability": values[start],
        "stateValues": values,
        "optimalActions": optimal_actions,
        "certificates": certificates,
        "nonTerminatingStates": non_terminating_states_interval(
            states, terminals, actions),
    }
