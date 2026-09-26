"""Unit tests for the exact rational reachability solver."""

import unittest
from fractions import Fraction as F

from app.solver import (_canonical_worst_distribution, _simplex_min,
                        non_terminating_states, non_terminating_states_interval,
                        solve_reachability, solve_robust_reachability)


def model(states, start, rescued, lost, actions):
    return solve_reachability(states, start, rescued, lost, actions)


def robust_model(states, start, rescued, lost, actions):
    return solve_robust_reachability(states, start, rescued, lost, actions)


def pointwise(actions):
    """Turn an exact distribution model into the equivalent point-interval
    (degenerate) robust model."""
    return {s: {a: {t: (p, p) for t, p in dist.items()}
                for a, dist in state_actions.items()}
            for s, state_actions in actions.items()}


class SimplexTest(unittest.TestCase):
    def test_trivial_constraint(self):
        # min x s.t. x >= 1  ->  x = 1
        self.assertEqual(_simplex_min([F(1)], [[F(1)]], [F(1)]), [F(1)])

    def test_componentwise_minimum(self):
        # min x + y s.t. x - y >= 0, y >= 1  ->  (1, 1)
        x = _simplex_min([F(1), F(1)],
                         [[F(1), F(-1)], [F(0), F(1)]],
                         [F(0), F(1)])
        self.assertEqual(x, [F(1), F(1)])

    def test_zero_solution(self):
        # min x + y s.t. x - y >= 0, y - x >= 0  ->  (0, 0)
        x = _simplex_min([F(1), F(1)],
                         [[F(1), F(-1)], [F(-1), F(1)]],
                         [F(0), F(0)])
        self.assertEqual(x, [F(0), F(0)])

    def test_no_constraints(self):
        self.assertEqual(_simplex_min([F(1), F(1)], [], []), [F(0), F(0)])

    def test_less_equal_row(self):
        # min -x s.t. x <= 1  ->  x = 1  (a '<=' row, i.e. -x >= -1)
        self.assertEqual(_simplex_min([F(-1)], [[F(-1)]], [F(-1)]), [F(1)])

    def test_mixed_senses(self):
        # min x + y s.t. x >= 1/2, x + y <= 2  ->  (1/2, 0)
        x = _simplex_min([F(1), F(1)],
                         [[F(1), F(0)], [F(-1), F(-1)]],
                         [F(1, 2), F(-2)])
        self.assertEqual(x, [F(1, 2), F(0)])


class CertainRescueTest(unittest.TestCase):
    """A model where rescue is certain: exact value 1 via 1/2 + 1/2."""

    def setUp(self):
        self.result = model(
            states=["s0", "s1", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {"push": {"s1": F(1, 2), "rescued": F(1, 2)}},
                "s1": {"push": {"rescued": F(1)}},
            },
        )

    def test_exact_probability_one(self):
        self.assertEqual(self.result["maxRescueProbability"], F(1))
        self.assertEqual(self.result["stateValues"]["s0"], F(1))
        self.assertEqual(self.result["stateValues"]["s1"], F(1))
        self.assertEqual(self.result["stateValues"]["rescued"], F(1))
        self.assertEqual(self.result["stateValues"]["lost"], F(0))

    def test_optimal_actions(self):
        self.assertEqual(self.result["optimalActions"],
                         {"s0": "push", "s1": "push"})

    def test_no_nonterminating_states(self):
        self.assertEqual(self.result["nonTerminatingStates"], [])


class ThirdsTest(unittest.TestCase):
    """1/3 and 2/3 transitions must yield exact thirds in the values."""

    def setUp(self):
        self.result = model(
            states=["s0", "s1", "s2", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {"commit": {"s1": F(2, 3), "lost": F(1, 3)}},
                "s1": {"retry": {"rescued": F(1, 2), "s1": F(1, 2)}},
                "s2": {"call": {"rescued": F(1, 3), "lost": F(2, 3)}},
            },
        )

    def test_exact_fractions(self):
        values = self.result["stateValues"]
        self.assertEqual(values["s1"], F(1))          # 1/2 + 1/2*v  =>  v = 1
        self.assertEqual(values["s0"], F(2, 3))       # 2/3 * 1
        self.assertEqual(values["s2"], F(1, 3))
        self.assertEqual(self.result["maxRescueProbability"], F(2, 3))

    def test_certificate_expected_values_exact(self):
        cert = {c["state"]: c for c in self.result["certificates"]}
        s2_actions = {a["action"]: a for a in cert["s2"]["actions"]}
        self.assertEqual(s2_actions["call"]["expectedValue"], F(1, 3))


class TieActionTest(unittest.TestCase):
    """Equal-value actions: canonical choice is the smallest identifier and
    the certificate shows exactly equal expected values."""

    def setUp(self):
        self.result = model(
            states=["s0", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {
                    "zulu": {"rescued": F(1, 2), "lost": F(1, 2)},
                    "alpha": {"rescued": F(1, 2), "lost": F(1, 2)},
                    "mike": {"rescued": F(1, 4), "lost": F(3, 4)},
                },
            },
        )

    def test_canonical_lexicographic_choice(self):
        self.assertEqual(self.result["optimalActions"]["s0"], "alpha")

    def test_certificate(self):
        (cert,) = self.result["certificates"]
        self.assertEqual(cert["state"], "s0")
        self.assertEqual(cert["value"], F(1, 2))
        self.assertEqual(cert["selectedAction"], "alpha")
        entries = {a["action"]: a for a in cert["actions"]}
        # equal expected values for the tied optimal actions
        self.assertEqual(entries["alpha"]["expectedValue"], F(1, 2))
        self.assertEqual(entries["zulu"]["expectedValue"], F(1, 2))
        self.assertTrue(entries["alpha"]["optimal"])
        self.assertTrue(entries["zulu"]["optimal"])
        self.assertTrue(entries["alpha"]["selected"])
        self.assertFalse(entries["zulu"]["selected"])
        # dominated action excluded with its (smaller) expectation visible
        self.assertEqual(entries["mike"]["expectedValue"], F(1, 4))
        self.assertFalse(entries["mike"]["optimal"])
        self.assertFalse(entries["mike"]["selected"])


class ClosedLoopTest(unittest.TestCase):
    """A closed loop that can never reach any terminal, plus a start state
    that is forced into it: zero rescue probability everywhere."""

    def setUp(self):
        self.result = model(
            states=["s0", "loop1", "loop2", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {"dive": {"loop1": F(1)}},
                "loop1": {"drift": {"loop2": F(1)}},
                "loop2": {"drift": {"loop1": F(1)}},
            },
        )

    def test_zero_rescue_probability(self):
        self.assertEqual(self.result["maxRescueProbability"], F(0))
        for s in ("s0", "loop1", "loop2"):
            self.assertEqual(self.result["stateValues"][s], F(0))

    def test_closed_loop_identified(self):
        self.assertEqual(self.result["nonTerminatingStates"],
                         ["loop1", "loop2", "s0"])

    def test_canonical_actions_still_reported(self):
        self.assertEqual(self.result["optimalActions"],
                         {"s0": "dive", "loop1": "drift", "loop2": "drift"})


class LoopWithExitTest(unittest.TestCase):
    """A loop with an exit to the lost terminal is NOT non-terminating, but
    its rescue probability is still exactly zero."""

    def test(self):
        result = model(
            states=["s0", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {
                    "spin": {"s0": F(1)},
                    "sink": {"lost": F(1)},
                },
            },
        )
        self.assertEqual(result["maxRescueProbability"], F(0))
        self.assertEqual(result["nonTerminatingStates"], [])
        # both actions are optimal (expected 0 == value 0); smallest id wins
        self.assertEqual(result["optimalActions"]["s0"], "sink")


class ChoiceTest(unittest.TestCase):
    """The solver must pick the action with the strictly larger exact
    expectation, including through multi-step paths."""

    def test_better_action_wins(self):
        result = model(
            states=["s0", "s1", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {
                    "direct": {"rescued": F(1, 3), "lost": F(2, 3)},
                    "via": {"s1": F(1)},
                },
                "s1": {"push": {"rescued": F(3, 4), "lost": F(1, 4)}},
            },
        )
        self.assertEqual(result["maxRescueProbability"], F(3, 4))
        self.assertEqual(result["optimalActions"]["s0"], "via")
        cert = {c["state"]: c for c in result["certificates"]}["s0"]
        entries = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(entries["direct"]["expectedValue"], F(1, 3))
        self.assertFalse(entries["direct"]["optimal"])
        self.assertEqual(entries["via"]["expectedValue"], F(3, 4))
        self.assertTrue(entries["via"]["optimal"])

    def test_start_on_terminal(self):
        result = model(
            states=["s0", "rescued"],
            start="rescued",
            rescued=["rescued"],
            lost=[],
            actions={"s0": {"go": {"rescued": F(1)}}},
        )
        self.assertEqual(result["maxRescueProbability"], F(1))


class NonTerminatingHelperTest(unittest.TestCase):
    def test_pure_graph(self):
        self.assertEqual(
            non_terminating_states(
                ["a", "b", "t"],
                {"t"},
                {"a": {"x": {"b": F(1)}}, "b": {"x": {"a": F(1)}}},
            ),
            ["a", "b"],
        )


class WorstDistributionTest(unittest.TestCase):
    """The greedy worst-vertex minimisation over an interval polytope."""

    def test_pours_remaining_mass_onto_lowest_values(self):
        bounds = {"a": (F(0), F(1, 2)), "b": (F(1, 4), F(1, 2)),
                  "c": (F(0), F(1, 2))}
        values = {"a": F(1, 10), "b": F(1, 2), "c": F(9, 10)}
        expected, dist = _canonical_worst_distribution(bounds, values)
        # lower bounds first (b = 1/4), then the 3/4 remainder goes to the
        # cheapest targets: a up to its cap, then b up to its cap, none to c
        self.assertEqual(dist, {"a": F(1, 2), "b": F(1, 2), "c": F(0)})
        self.assertEqual(expected, F(3, 10))
        self.assertEqual(sum(dist.values()), F(1))

    def test_respects_lower_bounds(self):
        bounds = {"x": (F(1, 2), F(1)), "y": (F(0), F(1, 2))}
        values = {"x": F(1), "y": F(0)}
        expected, dist = _canonical_worst_distribution(bounds, values)
        self.assertEqual(dist, {"x": F(1, 2), "y": F(1, 2)})
        self.assertEqual(expected, F(1, 2))

    def test_canonical_tie_break_by_target_identifier(self):
        bounds = {"y": (F(0), F(1)), "x": (F(0), F(1))}
        values = {"x": F(1, 2), "y": F(1, 2)}
        expected, dist = _canonical_worst_distribution(bounds, values)
        # equal values: the whole unit goes to the lexicographically first
        self.assertEqual(dist, {"x": F(1), "y": F(0)})
        self.assertEqual(expected, F(1, 2))


class RobustShiftableMassTest(unittest.TestCase):
    """Actions whose probability mass can shift between two successors must
    yield the exact lower bound, with the attaining worst distribution."""

    def test_mass_shift_between_rescued_and_lost(self):
        result = robust_model(
            states=["s0", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={"s0": {"commit": {"rescued": (F(1, 4), F(3, 4)),
                                       "lost": (F(1, 4), F(3, 4))}}},
        )
        self.assertEqual(result["maxRescueProbability"], F(1, 4))
        (entry,) = result["certificates"][0]["actions"]
        self.assertEqual(entry["expectedValue"], F(1, 4))
        self.assertEqual(entry["worstDistribution"],
                         {"rescued": F(1, 4), "lost": F(3, 4)})

    def test_mass_shift_between_two_controllable_successors(self):
        result = robust_model(
            states=["s0", "s1", "s2", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {"commit": {"s1": (F(1, 4), F(3, 4)),
                                  "s2": (F(1, 4), F(3, 4))}},
                "s1": {"push": {"rescued": (F(3, 4), F(3, 4)),
                                "lost": (F(1, 4), F(1, 4))}},
                "s2": {"call": {"rescued": (F(1, 4), F(1, 4)),
                                "lost": (F(3, 4), F(3, 4))}},
            },
        )
        # worst shift: 1/4 * 3/4 + 3/4 * 1/4 = 3/8 exactly
        self.assertEqual(result["maxRescueProbability"], F(3, 8))
        cert = {c["state"]: c for c in result["certificates"]}["s0"]
        (entry,) = cert["actions"]
        self.assertEqual(entry["expectedValue"], F(3, 8))
        self.assertEqual(entry["worstDistribution"],
                         {"s1": F(1, 4), "s2": F(3, 4)})


class RobustInfiniteHorizonTest(unittest.TestCase):
    """Infinite-horizon values must be exact rationals, not truncated
    finite-round approximations."""

    def test_geometric_loop_is_exactly_one(self):
        # worst case loops with 2/3 each round: sum_{k} (2/3)^k * 1/3 = 1
        result = robust_model(
            states=["s0", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={"s0": {"retry": {"rescued": (F(1, 3), F(1, 2)),
                                      "s0": (F(1, 2), F(2, 3))}}},
        )
        self.assertEqual(result["maxRescueProbability"], F(1))

    def test_interval_loop_exact_fraction(self):
        # worst vertex: rescued 1/4, lost 1/3, s0 5/12
        # v = 1/4 + (5/12) v  =>  v = 3/7 exactly
        result = robust_model(
            states=["s0", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={"s0": {"drift": {"rescued": (F(1, 4), F(1, 3)),
                                      "lost": (F(1, 4), F(1, 3)),
                                      "s0": (F(1, 3), F(1, 2))}}},
        )
        self.assertEqual(result["maxRescueProbability"], F(3, 7))
        (entry,) = result["certificates"][0]["actions"]
        self.assertEqual(entry["worstDistribution"],
                         {"rescued": F(1, 4), "lost": F(1, 3),
                          "s0": F(5, 12)})
        self.assertEqual(entry["expectedValue"], F(3, 7))


class RobustChoiceTest(unittest.TestCase):
    """The controller maximises the worst-case expectation; the certificate
    shows why the excluded action loses."""

    def setUp(self):
        self.result = robust_model(
            states=["s0", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {
                    "safe": {"rescued": (F(1, 2), F(3, 4)),
                             "lost": (F(1, 4), F(1, 2))},
                    "risky": {"rescued": (F(1, 3), F(1)),
                              "lost": (F(0), F(2, 3))},
                },
            },
        )

    def test_best_worst_case_wins(self):
        self.assertEqual(self.result["maxRescueProbability"], F(1, 2))
        self.assertEqual(self.result["optimalActions"], {"s0": "safe"})

    def test_certificate_explains_exclusion(self):
        entries = {a["action"]: a
                   for a in self.result["certificates"][0]["actions"]}
        self.assertEqual(entries["safe"]["expectedValue"], F(1, 2))
        self.assertTrue(entries["safe"]["optimal"])
        self.assertTrue(entries["safe"]["selected"])
        self.assertEqual(entries["risky"]["expectedValue"], F(1, 3))
        self.assertFalse(entries["risky"]["optimal"])
        self.assertFalse(entries["risky"]["selected"])
        self.assertEqual(entries["risky"]["worstDistribution"],
                         {"rescued": F(1, 3), "lost": F(2, 3)})

    def test_canonical_choice_on_interval_tie(self):
        result = robust_model(
            states=["s0", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {
                    "zulu": {"rescued": (F(1, 4), F(3, 4)),
                             "lost": (F(1, 4), F(3, 4))},
                    "alpha": {"rescued": (F(1, 4), F(3, 4)),
                              "lost": (F(1, 4), F(3, 4))},
                },
            },
        )
        self.assertEqual(result["optimalActions"]["s0"], "alpha")
        entries = {a["action"]: a
                   for a in result["certificates"][0]["actions"]}
        self.assertEqual(entries["alpha"]["worstDistribution"],
                         entries["zulu"]["worstDistribution"])


class RobustClosedLoopTest(unittest.TestCase):
    """Closed loops keep zero rescue probability under intervals too."""

    def test_interval_closed_loop_zero_and_identified(self):
        result = robust_model(
            states=["s0", "loop1", "loop2", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {"dive": {"loop1": (F(1), F(1))}},
                "loop1": {"drift": {"loop2": (F(1, 2), F(1)),
                                    "loop1": (F(0), F(1, 2))}},
                "loop2": {"drift": {"loop1": (F(1), F(1))}},
            },
        )
        self.assertEqual(result["maxRescueProbability"], F(0))
        for s in ("s0", "loop1", "loop2"):
            self.assertEqual(result["stateValues"][s], F(0))
        self.assertEqual(result["nonTerminatingStates"],
                         ["loop1", "loop2", "s0"])

    def test_adversary_can_hold_escapable_loop_forever(self):
        # the support reaches 'rescued' (upper 1/2 > 0), so the state is not
        # non-terminating, but the perturbation can keep all mass in the loop
        result = robust_model(
            states=["s0", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={"s0": {"spin": {"s0": (F(1, 2), F(1)),
                                     "rescued": (F(0), F(1, 2))}}},
        )
        self.assertEqual(result["maxRescueProbability"], F(0))
        self.assertEqual(result["nonTerminatingStates"], [])
        (entry,) = result["certificates"][0]["actions"]
        self.assertEqual(entry["worstDistribution"],
                         {"s0": F(1), "rescued": F(0)})


class RobustRegressionTest(unittest.TestCase):
    """Cases where naive relaxations of the max-min problem go wrong."""

    def test_forced_rescue_lower_bound(self):
        # rescued's lower bound 1/4 forces the adversary to expose the chain
        # to rescue every round; the exact guaranteed value is 1/4 (a
        # vertex-constraint relaxation of the Bellman inequation would
        # overestimate it)
        result = robust_model(
            states=["c0", "rescued", "lost"],
            start="c0",
            rescued=["rescued"],
            lost=["lost"],
            actions={"c0": {"a0": {"rescued": (F(1, 4), F(1, 2)),
                                   "c0": (F(0), F(1, 2)),
                                   "lost": (F(1, 2), F(1))}}},
        )
        self.assertEqual(result["maxRescueProbability"], F(1, 4))
        (entry,) = result["certificates"][0]["actions"]
        self.assertEqual(entry["expectedValue"], F(1, 4))
        self.assertEqual(entry["worstDistribution"],
                         {"rescued": F(1, 4), "c0": F(0), "lost": F(3, 4)})

    def test_adversary_loop_beyond_one_step_greedy(self):
        # the adversary's best reply loops x <-> y; a one-step greedy check
        # against a flat value vector would wrongly keep the exit vertex and
        # report 1 instead of 1/2
        result = robust_model(
            states=["x", "y", "rescued", "lost"],
            start="x",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "x": {"a": {"y": (F(1), F(1))},
                      "b": {"rescued": (F(1, 2), F(1, 2)),
                            "lost": (F(1, 2), F(1, 2))}},
                "y": {"c": {"x": (F(0), F(1)), "rescued": (F(0), F(1))}},
            },
        )
        self.assertEqual(result["maxRescueProbability"], F(1, 2))
        self.assertEqual(result["stateValues"]["x"], F(1, 2))
        self.assertEqual(result["stateValues"]["y"], F(1, 2))
        # both of x's actions are optimal; the canonical pick is the smallest
        self.assertEqual(result["optimalActions"]["x"], "a")
        cert = {c["state"]: c for c in result["certificates"]}["y"]
        (entry,) = cert["actions"]
        self.assertEqual(entry["worstDistribution"],
                         {"x": F(1), "rescued": F(0)})

    def test_controller_switch_improves(self):
        # the initial strategy (lexicographically first action) is
        # suboptimal; strategy iteration must switch to the better one
        result = robust_model(
            states=["s0", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {
                    "a_first": {"rescued": (F(1, 4), F(1, 4)),
                                "lost": (F(3, 4), F(3, 4))},
                    "b_better": {"rescued": (F(1, 2), F(3, 4)),
                                 "lost": (F(1, 4), F(1, 2))},
                },
            },
        )
        self.assertEqual(result["maxRescueProbability"], F(1, 2))
        self.assertEqual(result["optimalActions"], {"s0": "b_better"})


class RobustDegenerateConsistencyTest(unittest.TestCase):
    """Point intervals (lower == upper) degenerate to exact probabilities:
    the robust result must agree with the plain audit item by item."""

    def assert_consistent(self, states, start, rescued, lost, actions):
        exact = model(states, start, rescued, lost, actions)
        robust = robust_model(states, start, rescued, lost,
                              pointwise(actions))
        self.assertEqual(exact["maxRescueProbability"],
                         robust["maxRescueProbability"])
        self.assertEqual(exact["stateValues"], robust["stateValues"])
        self.assertEqual(exact["optimalActions"], robust["optimalActions"])
        self.assertEqual(exact["nonTerminatingStates"],
                         robust["nonTerminatingStates"])
        self.assertEqual(len(exact["certificates"]),
                         len(robust["certificates"]))
        for e_cert, r_cert in zip(exact["certificates"],
                                  robust["certificates"]):
            self.assertEqual(e_cert["state"], r_cert["state"])
            self.assertEqual(e_cert["value"], r_cert["value"])
            self.assertEqual(e_cert["selectedAction"],
                             r_cert["selectedAction"])
            e_entries = {a["action"]: a for a in e_cert["actions"]}
            r_entries = {a["action"]: a for a in r_cert["actions"]}
            self.assertEqual(set(e_entries), set(r_entries))
            for action_id in e_entries:
                self.assertEqual(e_entries[action_id]["expectedValue"],
                                 r_entries[action_id]["expectedValue"])
                self.assertEqual(e_entries[action_id]["optimal"],
                                 r_entries[action_id]["optimal"])
                self.assertEqual(e_entries[action_id]["selected"],
                                 r_entries[action_id]["selected"])
                # the worst distribution degenerates to the exact one
                self.assertEqual(r_entries[action_id]["worstDistribution"],
                                 actions[r_cert["state"]][action_id])

    def test_thirds_model(self):
        self.assert_consistent(
            states=["s0", "s1", "s2", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {"commit": {"s1": F(2, 3), "lost": F(1, 3)}},
                "s1": {"retry": {"rescued": F(1, 2), "s1": F(1, 2)}},
                "s2": {"call": {"rescued": F(1, 3), "lost": F(2, 3)}},
            },
        )

    def test_tie_model(self):
        self.assert_consistent(
            states=["s0", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {
                    "zulu": {"rescued": F(1, 2), "lost": F(1, 2)},
                    "alpha": {"rescued": F(1, 2), "lost": F(1, 2)},
                    "mike": {"rescued": F(1, 4), "lost": F(3, 4)},
                },
            },
        )

    def test_closed_loop_model(self):
        self.assert_consistent(
            states=["s0", "loop1", "loop2", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {"dive": {"loop1": F(1)}},
                "loop1": {"drift": {"loop2": F(1)}},
                "loop2": {"drift": {"loop1": F(1)}},
            },
        )

    def test_choice_model(self):
        self.assert_consistent(
            states=["s0", "s1", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {
                    "direct": {"rescued": F(1, 3), "lost": F(2, 3)},
                    "via": {"s1": F(1)},
                },
                "s1": {"push": {"rescued": F(3, 4), "lost": F(1, 4)}},
            },
        )

    def test_loop_with_exit_model(self):
        self.assert_consistent(
            states=["s0", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {
                    "spin": {"s0": F(1)},
                    "sink": {"lost": F(1)},
                },
            },
        )


class NonTerminatingIntervalHelperTest(unittest.TestCase):
    def test_upper_bound_support(self):
        # 'b' can never receive mass from 'a' (a's lower bounds already sum
        # to 1), so the a->b edge is not in the support even though its upper
        # bound is positive; 'a' loops forever while 'b' itself reaches 't'
        self.assertEqual(
            non_terminating_states_interval(
                ["a", "b", "t"],
                {"t"},
                {"a": {"x": {"a": (F(1), F(1)), "b": (F(0), F(1, 2))}},
                 "b": {"x": {"t": (F(1), F(1))}}},
            ),
            ["a"],
        )

    def test_point_intervals_match_plain_support(self):
        self.assertEqual(
            non_terminating_states_interval(
                ["a", "b", "t"],
                {"t"},
                {"a": {"x": {"b": (F(1), F(1))}},
                 "b": {"x": {"a": (F(1), F(1))}}},
            ),
            ["a", "b"],
        )


if __name__ == "__main__":
    unittest.main()
