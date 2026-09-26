"""Unit tests for the exact rational reachability solver."""

import unittest
from fractions import Fraction as F

from app.solver import _simplex_min, non_terminating_states, solve_reachability


def model(states, start, rescued, lost, actions):
    return solve_reachability(states, start, rescued, lost, actions)


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


if __name__ == "__main__":
    unittest.main()
