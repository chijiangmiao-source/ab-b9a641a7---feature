"""Unit tests for the exact rational robust (interval-MDP) solver."""

import unittest
from fractions import Fraction as F

from app.robust_solver import (
    interval_distribution_feasible,
    solve_robust_reachability,
    worst_distribution,
)
from app.solver import solve_reachability


def intervals_from_exact(actions):
    return {
        state: {
            action: {target: (prob, prob) for target, prob in dist.items()}
            for action, dist in state_actions.items()
        }
        for state, state_actions in actions.items()
    }


class FeasibilityTest(unittest.TestCase):
    def test_sum_bounds(self):
        self.assertTrue(interval_distribution_feasible(
            {"a": (F(1), F(1))}))
        self.assertTrue(interval_distribution_feasible(
            {"a": (F(1, 2), F(1)), "b": (F(1, 2), F(1))}))
        self.assertFalse(interval_distribution_feasible(
            {"a": (F(0), F(1, 3)), "b": (F(0), F(1, 3))}))
        self.assertFalse(interval_distribution_feasible(
            {"a": (F(2, 3), F(1)), "b": (F(2, 3), F(1))}))
        self.assertFalse(interval_distribution_feasible(
            {"a": (F(1, 2), F(1, 3))}))
        self.assertFalse(interval_distribution_feasible({}))


class WorstDistributionTest(unittest.TestCase):
    def test_slack_goes_to_cheapest(self):
        dist, value = worst_distribution(
            ["hi", "lo"],
            {"hi": F(1, 4), "lo": F(1, 2)},
            {"hi": F(1, 2), "lo": F(3, 4)},
            {"hi": F(1), "lo": F(0)})
        self.assertEqual(dist, {"hi": F(1, 4), "lo": F(3, 4)})
        self.assertEqual(value, F(1, 4))

    def test_canonical_tie_break_is_lexicographic(self):
        dist, value = worst_distribution(
            ["b", "a"], {"b": F(0), "a": F(0)},
            {"b": F(1), "a": F(1)}, {"b": F(0), "a": F(0)})
        self.assertEqual(sum(dist.values()), F(1))
        self.assertEqual(dist, {"a": F(1), "b": F(0)})
        self.assertEqual(value, F(0))


class MassShiftTest(unittest.TestCase):
    """An action whose free mass can move between rescued and a doomed state:
    the adversary claims the exact lower bound on rescue."""

    def setUp(self):
        self.result = solve_robust_reachability(
            states=["s0", "s1", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {"go": {
                    "rescued": (F(1, 4), F(1, 2)),
                    "s1": (F(1, 2), F(3, 4))}},
                "s1": {"sink": {"lost": (F(1), F(1))}},
            })

    def test_exact_lower_bound(self):
        self.assertEqual(self.result["maxRescueProbability"], F(1, 4))
        self.assertEqual(self.result["stateValues"]["s1"], F(0))

    def test_worst_distribution(self):
        entry = self.result["certificates"][0]["actions"][0]
        self.assertEqual(entry["expectedValue"], F(1, 4))
        self.assertTrue(entry["optimal"] and entry["selected"])
        self.assertEqual(entry["worstDistribution"],
                         {"rescued": F(1, 4), "s1": F(3, 4)})
        # the reported distribution is feasible and normalised
        probs = entry["worstDistribution"]
        self.assertEqual(sum(probs.values()), F(1))
        for t, p in probs.items():
            self.assertTrue(entry["lower"][t] <= p <= entry["upper"][t])

    def test_full_shift_means_zero(self):
        result = solve_robust_reachability(
            states=["s0", "rescued", "lost"], start="s0",
            rescued=["rescued"], lost=["lost"],
            actions={"s0": {"go": {
                "rescued": (F(0), F(1)), "lost": (F(0), F(1))}}})
        self.assertEqual(result["maxRescueProbability"], F(0))
        self.assertEqual(
            result["certificates"][0]["actions"][0]["worstDistribution"],
            {"lost": F(1), "rescued": F(0)})

    def test_mandatory_mass_saturated_sum(self):
        # sum lower == 1: a zero-lower target is forced to zero despite cap 1
        result = solve_robust_reachability(
            states=["s0", "rescued", "lost"], start="s0",
            rescued=["rescued"], lost=["lost"],
            actions={"s0": {"go": {
                "rescued": (F(0), F(1)), "lost": (F(1), F(1))}}})
        self.assertEqual(result["maxRescueProbability"], F(0))
        self.assertEqual(
            result["certificates"][0]["actions"][0]["worstDistribution"],
            {"rescued": F(0), "lost": F(1)})


class DegenerateMatchesExactTest(unittest.TestCase):
    """Point intervals [p,p] must reproduce the ordinary audit exactly."""

    MODELS = [
        ("thirds",
         ["s0", "s1", "s2", "rescued", "lost"], "s0",
         ["rescued"], ["lost"],
         {"s0": {"commit": {"s1": F(2, 3), "lost": F(1, 3)}},
          "s1": {"retry": {"rescued": F(1, 2), "s1": F(1, 2)}},
          "s2": {"call": {"rescued": F(1, 3), "lost": F(2, 3)}}}),
        ("ties",
         ["s0", "rescued", "lost"], "s0", ["rescued"], ["lost"],
         {"s0": {
             "zulu": {"rescued": F(1, 2), "lost": F(1, 2)},
             "alpha": {"rescued": F(1, 2), "lost": F(1, 2)},
             "mike": {"rescued": F(1, 4), "lost": F(3, 4)}}}),
        ("loop-with-exit",
         ["s0", "rescued", "lost"], "s0", ["rescued"], ["lost"],
         {"s0": {"spin": {"s0": F(1)}, "sink": {"lost": F(1)}}}),
        ("certain",
         ["s0", "s1", "rescued", "lost"], "s0", ["rescued"], ["lost"],
         {"s0": {"push": {"s1": F(1, 2), "rescued": F(1, 2)}},
          "s1": {"push": {"rescued": F(1)}}}),
    ]

    def test_all_fields_match(self):
        for name, states, start, rescued, lost, actions in self.MODELS:
            with self.subTest(model=name):
                exact = solve_reachability(
                    states, start, rescued, lost, actions)
                robust = solve_robust_reachability(
                    states, start, rescued, lost,
                    intervals_from_exact(actions))
                self.assertEqual(robust["maxRescueProbability"],
                                 exact["maxRescueProbability"])
                self.assertEqual(robust["stateValues"],
                                 exact["stateValues"])
                self.assertEqual(robust["optimalActions"],
                                 exact["optimalActions"])
                self.assertEqual(robust["nonTerminatingStates"],
                                 exact["nonTerminatingStates"])
                ecerts = {c["state"]: c for c in exact["certificates"]}
                rcerts = {c["state"]: c for c in robust["certificates"]}
                self.assertEqual(set(ecerts), set(rcerts))
                for s, ec in ecerts.items():
                    rc = rcerts[s]
                    self.assertEqual(rc["value"], ec["value"])
                    self.assertEqual(rc["selectedAction"],
                                     ec["selectedAction"])
                    ee = {a["action"]: a for a in ec["actions"]}
                    re_ = {a["action"]: a for a in rc["actions"]}
                    self.assertEqual(set(ee), set(re_))
                    for aid, ea in ee.items():
                        ra = re_[aid]
                        self.assertEqual(ra["expectedValue"],
                                         ea["expectedValue"])
                        self.assertEqual(ra["optimal"], ea["optimal"])
                        self.assertEqual(ra["selected"], ea["selected"])
                        # worst distribution is the unique point interval one
                        self.assertEqual(
                            sum(ra["worstDistribution"].values()), F(1))


class ClosedLoopRobustTest(unittest.TestCase):
    def setUp(self):
        self.result = solve_robust_reachability(
            states=["s0", "loop1", "loop2", "rescued", "lost"],
            start="s0", rescued=["rescued"], lost=["lost"],
            actions={
                "s0": {"dive": {"loop1": (F(1), F(1))}},
                "loop1": {"drift": {"loop2": (F(1), F(1))}},
                "loop2": {"drift": {"loop1": (F(1), F(1))}},
            })

    def test_zero(self):
        self.assertEqual(self.result["maxRescueProbability"], F(0))

    def test_nonterminating(self):
        self.assertEqual(self.result["nonTerminatingStates"],
                         ["loop1", "loop2", "s0"])

    def test_escape_edge_present_in_union_does_not_trap(self):
        # the interval allows positive escape mass -> not non-terminating,
        # but the adversary still pushes escape to zero: value stays zero
        result = solve_robust_reachability(
            states=["s0", "rescued", "lost"], start="s0",
            rescued=["rescued"], lost=["lost"],
            actions={"s0": {"go": {
                "s0": (F(1, 2), F(1)),
                "rescued": (F(0), F(1, 2))}}})
        self.assertEqual(result["maxRescueProbability"], F(0))
        self.assertEqual(result["nonTerminatingStates"], [])

    def test_forced_loop_with_illusory_cap_is_nonterminating(self):
        # mandatory looping mass is already 1; rescue cap can never be used
        result = solve_robust_reachability(
            states=["s0", "rescued", "lost"], start="s0",
            rescued=["rescued"], lost=["lost"],
            actions={"s0": {"go": {
                "s0": (F(1), F(1)), "rescued": (F(0), F(1))}}})
        self.assertEqual(result["nonTerminatingStates"], ["s0"])
        self.assertEqual(result["maxRescueProbability"], F(0) )


class ControllerChoiceTest(unittest.TestCase):
    def test_safe_exact_action_beats_robust_zero_action(self):
        result = solve_robust_reachability(
            states=["s0", "s1", "trap", "rescued", "lost"], start="s0",
            rescued=["rescued"], lost=["lost"],
            actions={
                "s0": {
                    "risk": {"s1": (F(1), F(1))},
                    "safe": {"rescued": (F(3, 4), F(3, 4)),
                             "lost": (F(1, 4), F(1, 4))}},
                "s1": {"go": {"trap": (F(0), F(1)),
                              "rescued": (F(0), F(1))}},
                "trap": {"spin": {"trap": (F(1), F(1))}},
            })
        self.assertEqual(result["maxRescueProbability"], F(3, 4))
        self.assertEqual(result["optimalActions"]["s0"], "safe")
        cert = {c["state"]: c for c in result["certificates"]}["s0"]
        entries = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(entries["risk"]["expectedValue"], F(0))
        self.assertFalse(entries["risk"]["optimal"])
        self.assertEqual(entries["safe"]["expectedValue"], F(3, 4))
        self.assertTrue(entries["safe"]["selected"])

    def test_mandatory_rescue_through_selfloop_is_certain(self):
        # adversary delays maximally (loop 4/5, rescue 1/5 each visit),
        # geometric sum is exactly (1/5)/(1-4/5) = 1
        result = solve_robust_reachability(
            states=["s0", "rescued"], start="s0",
            rescued=["rescued"], lost=[],
            actions={"s0": {"r": {
                "rescued": (F(1, 5), F(1)), "s0": (F(0), F(4, 5))}}})
        self.assertEqual(result["maxRescueProbability"], F(1))
        entry = result["certificates"][0]["actions"][0]
        self.assertEqual(entry["expectedValue"], F(1))

    def test_fractional_value_through_loop(self):
        # slack mass lands on the loop state, whose exact value is 1/2
        result = solve_robust_reachability(
            states=["s0", "s1", "rescued", "lost"], start="s0",
            rescued=["rescued"], lost=["lost"],
            actions={
                "s0": {"go": {
                    "rescued": (F(1, 3), F(2, 3)),
                    "s1": (F(1, 3), F(2, 3))}},
                "s1": {"x": {"rescued": (F(1, 2), F(1, 2)),
                             "lost": (F(1, 2), F(1, 2))}}})
        self.assertEqual(result["maxRescueProbability"], F(2, 3))
        self.assertEqual(
            result["certificates"][0]["actions"][0]["worstDistribution"],
            {"rescued": F(1, 3), "s1": F(2, 3)})


class StrategyIterationTest(unittest.TestCase):
    def test_adversary_switches_after_value_refinement(self):
        # Against the pessimistic baseline both successors of s1 tie at 0;
        # once their values are known the adversary must move all mass to the
        # closed trap, and the start value drops below 1/2.
        result = solve_robust_reachability(
            states=["s0", "s1", "good", "trap", "rescued", "lost"],
            start="s0",
            rescued=["rescued"], lost=["lost"],
            actions={
                "s0": {"go": {"s1": (F(1), F(1))}},
                "s1": {"r": {"good": (F(0), F(1)), "trap": (F(0), F(1))}},
                "good": {"x": {"rescued": (F(1, 2), F(1, 2)),
                               "lost": (F(1, 2), F(1, 2))}},
                "trap": {"spin": {"trap": (F(1), F(1))}},
            })
        self.assertEqual(result["stateValues"]["good"], F(1, 2))
        self.assertEqual(result["stateValues"]["trap"], F(0))
        self.assertEqual(result["stateValues"]["s1"], F(0))
        self.assertEqual(result["maxRescueProbability"], F(0))


if __name__ == "__main__":
    unittest.main()
