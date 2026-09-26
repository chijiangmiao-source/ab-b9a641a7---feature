"""HTTP/API tests: exact fraction responses, certificates and every
locatable validation failure (which must never yield a success certificate)."""

import json
import threading
import unittest
import urllib.error
import urllib.request

from app.server import AUDIT_PATH, ROBUST_AUDIT_PATH, AuditHandler
from http.server import ThreadingHTTPServer


def post(url, payload, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class ApiTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), AuditHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:%d" % cls.port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def url(self, path):
        return self.base + path

    def assert_failure(self, payload, code, path_prefix=None, raw=None,
                       path=AUDIT_PATH):
        status, body = post(self.url(path), payload, raw=raw)
        self.assertEqual(status, 400, body)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], code, body)
        self.assertNotIn("certificates", body)
        self.assertNotIn("maxRescueProbability", body)
        if path_prefix is not None:
            self.assertIsNotNone(body["error"]["path"])
            self.assertTrue(str(body["error"]["path"]).startswith(path_prefix),
                            body)
        return body


class ApiTest(ApiTestBase):

    # -- health -------------------------------------------------------------
    def test_health(self):
        with urllib.request.urlopen(self.url("/health")) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(resp.status, 200)
        self.assertTrue(body["ok"])

    # -- success cases --------------------------------------------------------
    def test_certain_rescue_exact(self):
        status, body = post(self.url("/api/reachability-audit"), {
            "states": ["s0", "s1", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {
                "s0": {"push": {"s1": "1/2", "rescued": "1/2"}},
                "s1": {"push": {"rescued": "1"}},
            },
        })
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["maxRescueProbability"], "1")
        self.assertEqual(body["stateValues"]["s0"], "1")
        self.assertEqual(body["nonTerminatingStates"], [])

    def test_thirds_exact(self):
        status, body = post(self.url("/api/reachability-audit"), {
            "states": ["s0", "s1", "s2", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {
                "s0": {"commit": {"s1": "2/3", "lost": "1/3"}},
                "s1": {"retry": {"rescued": "1/2", "s1": "1/2"}},
                "s2": {"call": {"rescued": "1/3", "lost": "2/3"}},
            },
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["maxRescueProbability"], "2/3")
        self.assertEqual(body["stateValues"]["s2"], "1/3")
        self.assertEqual(body["stateValues"]["rescued"], "1")
        self.assertEqual(body["stateValues"]["lost"], "0")

    def test_tie_actions_canonical_and_certificate(self):
        status, body = post(self.url("/api/reachability-audit"), {
            "states": ["s0", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {
                "s0": {
                    "zulu": {"rescued": "1/2", "lost": "1/2"},
                    "alpha": {"rescued": "1/2", "lost": "1/2"},
                    "mike": {"rescued": "1/4", "lost": "3/4"},
                },
            },
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["optimalActions"]["s0"], "alpha")
        (cert,) = body["certificates"]
        self.assertEqual(cert["state"], "s0")
        self.assertEqual(cert["value"], "1/2")
        entries = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(entries["alpha"]["expectedValue"], "1/2")
        self.assertEqual(entries["zulu"]["expectedValue"], "1/2")
        self.assertTrue(entries["alpha"]["selected"])
        self.assertFalse(entries["zulu"]["selected"])
        self.assertEqual(entries["mike"]["expectedValue"], "1/4")
        self.assertFalse(entries["mike"]["optimal"])

    def test_closed_loop_zero_rescue(self):
        status, body = post(self.url("/api/reachability-audit"), {
            "states": ["s0", "loop1", "loop2", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {
                "s0": {"dive": {"loop1": "1"}},
                "loop1": {"drift": {"loop2": "1"}},
                "loop2": {"drift": {"loop1": "1"}},
            },
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["maxRescueProbability"], "0")
        self.assertEqual(body["stateValues"]["s0"], "0")
        self.assertEqual(body["stateValues"]["loop1"], "0")
        self.assertEqual(body["stateValues"]["loop2"], "0")
        self.assertEqual(body["nonTerminatingStates"], ["loop1", "loop2", "s0"])

    # -- validation failures ----------------------------------------------------
    def base_model(self):
        return {
            "states": ["s0", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {"s0": {"go": {"rescued": "1"}}},
        }

    def test_unknown_transition_target(self):
        m = self.base_model()
        m["actions"]["s0"]["go"] = {"rescued": "1/2", "s9": "1/2"}
        self.assert_failure(m, "UNKNOWN_STATE_REFERENCE", "actions.s0.go.s9")

    def test_unknown_start(self):
        m = self.base_model()
        m["start"] = "nowhere"
        self.assert_failure(m, "UNKNOWN_STATE_REFERENCE", "start")

    def test_unknown_terminal(self):
        m = self.base_model()
        m["rescuedStates"] = ["safe"]
        self.assert_failure(m, "UNKNOWN_STATE_REFERENCE", "rescuedStates[0]")

    def test_unknown_action_state(self):
        m = self.base_model()
        m["actions"]["ghost"] = {"go": {"rescued": "1"}}
        self.assert_failure(m, "UNKNOWN_STATE_REFERENCE", "actions.ghost")

    def test_nonterminal_missing_actions(self):
        m = self.base_model()
        m["states"].append("s1")
        self.assert_failure(m, "NONTERMINAL_MISSING_ACTIONS", "actions.s1")

    def test_nonterminal_empty_actions(self):
        m = self.base_model()
        m["states"].append("s1")
        m["actions"]["s1"] = {}
        self.assert_failure(m, "NONTERMINAL_MISSING_ACTIONS", "actions.s1")

    def test_terminal_with_actions(self):
        m = self.base_model()
        m["actions"]["rescued"] = {"go": {"lost": "1"}}
        self.assert_failure(m, "TERMINAL_HAS_ACTIONS", "actions.rescued")

    def test_probability_sum_not_one(self):
        m = self.base_model()
        m["actions"]["s0"]["go"] = {"rescued": "1/3", "lost": "1/3"}
        self.assert_failure(m, "PROBABILITY_SUM_INVALID", "actions.s0.go")

    def test_probability_sum_empty_distribution(self):
        m = self.base_model()
        m["actions"]["s0"]["go"] = {}
        self.assert_failure(m, "PROBABILITY_SUM_INVALID", "actions.s0.go")

    def test_invalid_fraction_format(self):
        m = self.base_model()
        m["actions"]["s0"]["go"] = {"rescued": "half"}
        self.assert_failure(m, "INVALID_FRACTION", "actions.s0.go.rescued")

    def test_float_probability_rejected(self):
        m = self.base_model()
        m["actions"]["s0"]["go"] = {"rescued": 0.5, "lost": 0.5}
        self.assert_failure(m, "INVALID_FRACTION", "actions.s0.go.rescued")

    def test_probability_out_of_range(self):
        m = self.base_model()
        m["actions"]["s0"]["go"] = {"rescued": "3/2", "lost": "-1/2"}
        self.assert_failure(m, "INVALID_PROBABILITY", "actions.s0.go.rescued")

    def test_duplicate_state(self):
        m = self.base_model()
        m["states"] = ["s0", "s0", "rescued", "lost"]
        self.assert_failure(m, "DUPLICATE_STATE", "states[1]")

    def test_terminal_conflict(self):
        m = self.base_model()
        m["lostStates"] = ["rescued", "lost"]
        self.assert_failure(m, "TERMINAL_CONFLICT")

    def test_duplicate_json_key(self):
        raw = (b'{"states": ["s0", "r"], "start": "s0", "rescuedStates": ["r"],'
               b' "lostStates": [], "actions": {"s0": {"a": {"r": "1"},'
               b' "a": {"r": "1"}}}}')
        self.assert_failure(None, "DUPLICATE_KEY", raw=raw)

    def test_invalid_json(self):
        self.assert_failure(None, "INVALID_JSON", raw=b"{not json")

    def test_missing_field(self):
        self.assert_failure({"states": ["s0"]}, "INVALID_SCHEMA", "start")

    def test_unknown_route_404(self):
        status, body = post(self.url("/nope"), {})
        self.assertEqual(status, 404)
        self.assertFalse(body["ok"])


def intervalise(actions):
    """Turn an exact-distribution actions mapping into the equivalent
    point-interval robust form."""
    return {s: {a: {t: {"lower": p, "upper": p} for t, p in dist.items()}
                for a, dist in state_actions.items()}
            for s, state_actions in actions.items()}


class RobustApiTest(ApiTestBase):
    """POST /api/robust-reachability-audit: interval-valued transitions,
    exact max-min rescue probability, canonical worst distributions."""

    def robust_base_model(self):
        return {
            "states": ["s0", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {"s0": {"go": {"rescued": {"lower": "1",
                                                  "upper": "1"}}}},
        }

    # -- success cases --------------------------------------------------------
    def test_shiftable_mass_exact_lower_bound(self):
        status, body = post(self.url(ROBUST_AUDIT_PATH), {
            "states": ["s0", "s1", "s2", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {
                "s0": {"commit": {"s1": {"lower": "1/4", "upper": "3/4"},
                                  "s2": {"lower": "1/4", "upper": "3/4"}}},
                "s1": {"push": {"rescued": {"lower": "3/4", "upper": "3/4"},
                                "lost": {"lower": "1/4", "upper": "1/4"}}},
                "s2": {"call": {"rescued": {"lower": "1/4", "upper": "1/4"},
                                "lost": {"lower": "3/4", "upper": "3/4"}}},
            },
        })
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        # worst shift of the free 1/2 onto s2: 1/4*3/4 + 3/4*1/4 = 3/8
        self.assertEqual(body["maxRescueProbability"], "3/8")
        cert = {c["state"]: c for c in body["certificates"]}["s0"]
        (entry,) = cert["actions"]
        self.assertEqual(entry["expectedValue"], "3/8")
        self.assertEqual(entry["worstDistribution"],
                         {"s1": "1/4", "s2": "3/4"})
        self.assertTrue(entry["optimal"])
        self.assertTrue(entry["selected"])

    def test_degenerate_intervals_match_plain_audit_item_by_item(self):
        actions = {
            "s0": {"commit": {"s1": "2/3", "lost": "1/3"}},
            "s1": {"retry": {"rescued": "1/2", "s1": "1/2"}},
            "s2": {"call": {"rescued": "1/3", "lost": "2/3"}},
        }
        base = {
            "states": ["s0", "s1", "s2", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
        }
        status_exact, exact = post(self.url(AUDIT_PATH),
                                   dict(base, actions=actions))
        status_robust, robust = post(
            self.url(ROBUST_AUDIT_PATH),
            dict(base, actions=intervalise(actions)))
        self.assertEqual(status_exact, 200)
        self.assertEqual(status_robust, 200)
        # every shared item must agree; the robust reply only adds the
        # (degenerate) worst distribution per action
        stripped = dict(robust)
        stripped["certificates"] = []
        for cert in robust["certificates"]:
            stripped["certificates"].append({
                **{k: v for k, v in cert.items()},
                "actions": [{k: v for k, v in entry.items()
                             if k != "worstDistribution"}
                            for entry in cert["actions"]],
            })
        self.assertEqual(stripped, exact)
        # and the degenerate worst distributions equal the exact ones
        for cert in robust["certificates"]:
            for entry in cert["actions"]:
                self.assertEqual(
                    entry["worstDistribution"],
                    actions[cert["state"]][entry["action"]])

    def test_interval_closed_loop_zero_rescue(self):
        status, body = post(self.url(ROBUST_AUDIT_PATH), {
            "states": ["s0", "loop1", "loop2", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {
                "s0": {"dive": {"loop1": {"lower": "1", "upper": "1"}}},
                "loop1": {"drift": {"loop2": {"lower": "1/2", "upper": "1"},
                                    "loop1": {"lower": "0", "upper": "1/2"}}},
                "loop2": {"drift": {"loop1": {"lower": "1", "upper": "1"}}},
            },
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["maxRescueProbability"], "0")
        self.assertEqual(body["stateValues"]["loop1"], "0")
        self.assertEqual(body["stateValues"]["loop2"], "0")
        self.assertEqual(body["nonTerminatingStates"],
                         ["loop1", "loop2", "s0"])

    def test_adversary_reselects_distribution_every_visit(self):
        # escapable loop (support reaches 'rescued') that the perturbation
        # can nevertheless hold forever: robust probability is exactly 0
        status, body = post(self.url(ROBUST_AUDIT_PATH), {
            "states": ["s0", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {"s0": {"spin": {"s0": {"lower": "1/2", "upper": "1"},
                                        "rescued": {"lower": "0",
                                                    "upper": "1/2"}}}},
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["maxRescueProbability"], "0")
        self.assertEqual(body["nonTerminatingStates"], [])
        (entry,) = body["certificates"][0]["actions"]
        self.assertEqual(entry["worstDistribution"],
                         {"rescued": "0", "s0": "1"})

    def test_geometric_loop_exact_one(self):
        status, body = post(self.url(ROBUST_AUDIT_PATH), {
            "states": ["s0", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {"s0": {"retry": {"rescued": {"lower": "1/3",
                                                     "upper": "1/2"},
                                         "s0": {"lower": "1/2",
                                                "upper": "2/3"}}}},
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["maxRescueProbability"], "1")

    def test_robust_tie_canonical_choice(self):
        status, body = post(self.url(ROBUST_AUDIT_PATH), {
            "states": ["s0", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {
                "s0": {
                    "zulu": {"rescued": {"lower": "1/4", "upper": "3/4"},
                             "lost": {"lower": "1/4", "upper": "3/4"}},
                    "alpha": {"rescued": {"lower": "1/4", "upper": "3/4"},
                              "lost": {"lower": "1/4", "upper": "3/4"}},
                },
            },
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["maxRescueProbability"], "1/4")
        self.assertEqual(body["optimalActions"]["s0"], "alpha")
        entries = {a["action"]: a for a in body["certificates"][0]["actions"]}
        self.assertEqual(entries["alpha"]["worstDistribution"],
                         {"lost": "3/4", "rescued": "1/4"})
        self.assertEqual(entries["alpha"]["worstDistribution"],
                         entries["zulu"]["worstDistribution"])

    # -- validation failures ----------------------------------------------------
    def assert_robust_failure(self, payload, code, path_prefix=None, raw=None):
        return self.assert_failure(payload, code, path_prefix, raw,
                                   path=ROBUST_AUDIT_PATH)

    def test_lower_bounds_sum_above_one(self):
        m = self.robust_base_model()
        m["actions"]["s0"]["go"] = {
            "rescued": {"lower": "2/3", "upper": "1"},
            "lost": {"lower": "2/3", "upper": "1"},
        }
        self.assert_robust_failure(m, "INTERVAL_SUM_INVALID", "actions.s0.go")

    def test_upper_bounds_sum_below_one(self):
        m = self.robust_base_model()
        m["actions"]["s0"]["go"] = {
            "rescued": {"lower": "0", "upper": "1/3"},
            "lost": {"lower": "0", "upper": "1/3"},
        }
        self.assert_robust_failure(m, "INTERVAL_SUM_INVALID", "actions.s0.go")

    def test_empty_interval_distribution(self):
        m = self.robust_base_model()
        m["actions"]["s0"]["go"] = {}
        self.assert_robust_failure(m, "INTERVAL_SUM_INVALID", "actions.s0.go")

    def test_lower_above_upper_located_at_target(self):
        m = self.robust_base_model()
        m["actions"]["s0"]["go"] = {
            "rescued": {"lower": "2/3", "upper": "1/3"},
            "lost": {"lower": "1/3", "upper": "1/3"},
        }
        self.assert_robust_failure(m, "INTERVAL_EMPTY",
                                   "actions.s0.go.rescued")

    def test_missing_upper_bound(self):
        m = self.robust_base_model()
        m["actions"]["s0"]["go"] = {"rescued": {"lower": "1"}}
        self.assert_robust_failure(m, "INVALID_SCHEMA",
                                   "actions.s0.go.rescued")

    def test_unexpected_interval_key(self):
        m = self.robust_base_model()
        m["actions"]["s0"]["go"] = {
            "rescued": {"lower": "1", "upper": "1", "mid": "1"},
        }
        self.assert_robust_failure(m, "INVALID_SCHEMA",
                                   "actions.s0.go.rescued")

    def test_interval_not_an_object(self):
        m = self.robust_base_model()
        m["actions"]["s0"]["go"] = {"rescued": "1"}
        self.assert_robust_failure(m, "INVALID_SCHEMA",
                                   "actions.s0.go.rescued")

    def test_float_bound_rejected(self):
        m = self.robust_base_model()
        m["actions"]["s0"]["go"] = {
            "rescued": {"lower": 0.5, "upper": "1"},
            "lost": {"lower": "0", "upper": "1/2"},
        }
        self.assert_robust_failure(m, "INVALID_FRACTION",
                                   "actions.s0.go.rescued.lower")

    def test_bound_out_of_range(self):
        m = self.robust_base_model()
        m["actions"]["s0"]["go"] = {
            "rescued": {"lower": "3/2", "upper": "2"},
            "lost": {"lower": "0", "upper": "1"},
        }
        self.assert_robust_failure(m, "INVALID_PROBABILITY",
                                   "actions.s0.go.rescued.lower")

    def test_invalid_bound_format(self):
        m = self.robust_base_model()
        m["actions"]["s0"]["go"] = {
            "rescued": {"lower": "half", "upper": "1"},
        }
        self.assert_robust_failure(m, "INVALID_FRACTION",
                                   "actions.s0.go.rescued.lower")

    def test_unknown_interval_target(self):
        m = self.robust_base_model()
        m["actions"]["s0"]["go"] = {
            "rescued": {"lower": "1/2", "upper": "1/2"},
            "s9": {"lower": "1/2", "upper": "1/2"},
        }
        self.assert_robust_failure(m, "UNKNOWN_STATE_REFERENCE",
                                   "actions.s0.go.s9")

    def test_terminal_with_actions(self):
        m = self.robust_base_model()
        m["actions"]["rescued"] = {"stay": {"rescued": {"lower": "1",
                                                        "upper": "1"}}}
        self.assert_robust_failure(m, "TERMINAL_HAS_ACTIONS",
                                   "actions.rescued")

    def test_nonterminal_missing_actions(self):
        m = self.robust_base_model()
        m["states"].append("s1")
        self.assert_robust_failure(m, "NONTERMINAL_MISSING_ACTIONS",
                                   "actions.s1")

    def test_unknown_start(self):
        m = self.robust_base_model()
        m["start"] = "nowhere"
        self.assert_robust_failure(m, "UNKNOWN_STATE_REFERENCE", "start")


if __name__ == "__main__":
    unittest.main()
