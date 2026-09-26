"""HTTP/API tests for the robust (interval-MDP) audit endpoint: exact
fraction answers, canonical worst distributions, degenerate equality with the
ordinary audit, closed-loop behaviour and locatable validation failures."""

import json
import threading
import unittest
import urllib.error
import urllib.request

from app.server import AuditHandler
from http.server import ThreadingHTTPServer

ROBUST_PATH = "/api/robust-reachability-audit"
EXACT_PATH = "/api/reachability-audit"


def post(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def interval_model(exact_payload):
    """Turn every 'p/q' transition body into {'lower','upper'} point boxes."""
    robust = json.loads(json.dumps(exact_payload))
    for state_actions in robust["actions"].values():
        for dist in state_actions.values():
            for target, prob in list(dist.items()):
                dist[target] = {"lower": prob, "upper": prob}
    return robust


class RobustApiTest(unittest.TestCase):
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

    # -- success cases -------------------------------------------------------
    def test_mass_shift_exact_lower_bound(self):
        status, body = post(self.url(ROBUST_PATH), {
            "states": ["s0", "s1", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {
                "s0": {"go": {
                    "rescued": {"lower": "1/4", "upper": "1/2"},
                    "s1": {"lower": "1/2", "upper": "3/4"}}},
                "s1": {"sink": {"lost": {"lower": "1", "upper": "1"}}},
            },
        })
        self.assertEqual(status, 200, body)
        self.assertTrue(body["ok"])
        self.assertEqual(body["maxRescueProbability"], "1/4")
        self.assertEqual(body["stateValues"]["s1"], "0")
        cert = {c["state"]: c for c in body["certificates"]}["s0"]
        (entry,) = cert["actions"]
        self.assertEqual(entry["expectedValue"], "1/4")
        self.assertTrue(entry["optimal"] and entry["selected"])
        wd = entry["worstDistribution"]
        self.assertEqual(wd["rescued"]["probability"], "1/4")
        self.assertEqual(wd["s1"]["probability"], "3/4")
        # endpoints echoed for recomputation
        self.assertEqual(wd["rescued"]["lower"], "1/4")
        self.assertEqual(wd["rescued"]["upper"], "1/2")

    def test_closed_loop_zero_and_nonterminating(self):
        status, body = post(self.url(ROBUST_PATH), {
            "states": ["s0", "loop1", "loop2", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {
                "s0": {"dive": {"loop1": {"lower": "1", "upper": "1"}}},
                "loop1": {"drift": {"loop2": {"lower": "1", "upper": "1"}}},
                "loop2": {"drift": {"loop1": {"lower": "1", "upper": "1"}}},
            },
        })
        self.assertEqual(status, 200, body)
        self.assertEqual(body["maxRescueProbability"], "0")
        self.assertEqual(body["nonTerminatingStates"],
                         ["loop1", "loop2", "s0"])

    def test_fractional_robust_value(self):
        status, body = post(self.url(ROBUST_PATH), {
            "states": ["s0", "s1", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {
                "s0": {"go": {
                    "rescued": {"lower": "1/3", "upper": "2/3"},
                    "s1": {"lower": "1/3", "upper": "2/3"}}},
                "s1": {"x": {
                    "rescued": {"lower": "1/2", "upper": "1/2"},
                    "lost": {"lower": "1/2", "upper": "1/2"}}},
            },
        })
        self.assertEqual(status, 200, body)
        self.assertEqual(body["maxRescueProbability"], "2/3")

    def test_certificate_shows_excluded_action_reason(self):
        status, body = post(self.url(ROBUST_PATH), {
            "states": ["s0", "s1", "trap", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {
                "s0": {
                    "risk": {"s1": {"lower": "1", "upper": "1"}},
                    "safe": {
                        "rescued": {"lower": "3/4", "upper": "3/4"},
                        "lost": {"lower": "1/4", "upper": "1/4"}}},
                "s1": {"go": {
                    "trap": {"lower": "0", "upper": "1"},
                    "rescued": {"lower": "0", "upper": "1"}}},
                "trap": {"spin": {"trap": {"lower": "1", "upper": "1"}}},
            },
        })
        self.assertEqual(status, 200, body)
        self.assertEqual(body["maxRescueProbability"], "3/4")
        self.assertEqual(body["optimalActions"]["s0"], "safe")
        cert = {c["state"]: c for c in body["certificates"]}["s0"]
        entries = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(entries["risk"]["expectedValue"], "0")
        self.assertFalse(entries["risk"]["selected"])
        # risk leads deterministically to s1 ...
        self.assertEqual(
            entries["risk"]["worstDistribution"]["s1"]["probability"], "1")
        self.assertEqual(entries["safe"]["expectedValue"], "3/4")
        self.assertTrue(entries["safe"]["selected"])
        # ... where the disturbance dumps all mass into the trap
        s1_cert = {c["state"]: c for c in body["certificates"]}["s1"]
        self.assertEqual(
            s1_cert["actions"][0]["worstDistribution"]["trap"]["probability"],
            "1")

    def test_degenerate_intervals_match_exact_audit_item_by_item(self):
        exact = {
            "states": ["s0", "s1", "s2", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {
                "s0": {"commit": {"s1": "2/3", "lost": "1/3"}},
                "s1": {"retry": {"rescued": "1/2", "s1": "1/2"}},
                "s2": {"call": {"rescued": "1/3", "lost": "2/3"}},
            },
        }
        s1, b1 = post(self.url(EXACT_PATH), exact)
        s2, b2 = post(self.url(ROBUST_PATH), interval_model(exact))
        self.assertEqual((s1, s2), (200, 200))
        for field in ("maxRescueProbability", "stateValues",
                      "optimalActions", "nonTerminatingStates", "start"):
            self.assertEqual(b1[field], b2[field], field)
        c1 = {c["state"]: c for c in b1["certificates"]}
        c2 = {c["state"]: c for c in b2["certificates"]}
        self.assertEqual(set(c1), set(c2))
        for state, cert in c1.items():
            self.assertEqual(c2[state]["value"], cert["value"])
            self.assertEqual(c2[state]["selectedAction"],
                             cert["selectedAction"])
            e1 = {a["action"]: a for a in cert["actions"]}
            e2 = {a["action"]: a for a in c2[state]["actions"]}
            for aid, a in e1.items():
                self.assertEqual(e2[aid]["expectedValue"],
                                 a["expectedValue"])
                self.assertEqual(e2[aid]["optimal"], a["optimal"])
                self.assertEqual(e2[aid]["selected"], a["selected"])

    # -- validation failures --------------------------------------------------
    def assert_failure(self, payload, code, path=None):
        status, body = post(self.url(ROBUST_PATH), payload)
        self.assertEqual(status, 400, body)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], code, body)
        self.assertNotIn("certificates", body)
        self.assertNotIn("maxRescueProbability", body)
        if path is not None:
            self.assertEqual(body["error"]["path"], path, body)
        return body

    def base_model(self):
        return {
            "states": ["s0", "rescued", "lost"],
            "start": "s0",
            "rescuedStates": ["rescued"],
            "lostStates": ["lost"],
            "actions": {"s0": {"go": {
                "rescued": {"lower": "0", "upper": "1"},
                "lost": {"lower": "0", "upper": "1"}}}},
        }

    def test_capacity_below_one(self):
        m = self.base_model()
        m["actions"]["s0"]["go"] = {
            "rescued": {"lower": "0", "upper": "1/3"},
            "lost": {"lower": "0", "upper": "1/3"}}
        self.assert_failure(m, "PROBABILITY_SUM_INVALID", "actions.s0.go")

    def test_mandatory_mass_above_one(self):
        m = self.base_model()
        m["actions"]["s0"]["go"] = {
            "rescued": {"lower": "2/3", "upper": "1"},
            "lost": {"lower": "2/3", "upper": "1"}}
        self.assert_failure(m, "PROBABILITY_SUM_INVALID", "actions.s0.go")

    def test_inverted_interval(self):
        m = self.base_model()
        m["actions"]["s0"]["go"] = {
            "rescued": {"lower": "1/2", "upper": "1/3"},
            "lost": {"lower": "1/2", "upper": "2/3"}}
        self.assert_failure(m, "INTERVAL_INVALID",
                            "actions.s0.go.rescued")

    def test_missing_bound(self):
        m = self.base_model()
        m["actions"]["s0"]["go"]["rescued"] = {"lower": "0"}
        self.assert_failure(m, "INTERVAL_INVALID",
                            "actions.s0.go.rescued")

    def test_interval_not_object(self):
        m = self.base_model()
        m["actions"]["s0"]["go"]["rescued"] = "1/2"
        self.assert_failure(m, "INTERVAL_INVALID",
                            "actions.s0.go.rescued")

    def test_float_bound_rejected(self):
        m = self.base_model()
        m["actions"]["s0"]["go"]["rescued"] = {"lower": 0.0, "upper": "1"}
        self.assert_failure(m, "INVALID_FRACTION",
                            "actions.s0.go.rescued.lower")

    def test_bound_out_of_range(self):
        m = self.base_model()
        m["actions"]["s0"]["go"]["rescued"] = {"lower": "0", "upper": "3/2"}
        self.assert_failure(m, "INVALID_PROBABILITY",
                            "actions.s0.go.rescued.upper")

    def test_unknown_target(self):
        m = self.base_model()
        m["actions"]["s0"]["go"] = {"s9": {"lower": "0", "upper": "1"}}
        self.assert_failure(m, "UNKNOWN_STATE_REFERENCE",
                            "actions.s0.go.s9")

    def test_empty_distribution(self):
        m = self.base_model()
        m["actions"]["s0"]["go"] = {}
        self.assert_failure(m, "PROBABILITY_SUM_INVALID", "actions.s0.go")

    def test_common_structure_failures_still_located(self):
        m = self.base_model()
        m["states"].append("s1")
        self.assert_failure(m, "NONTERMINAL_MISSING_ACTIONS", "actions.s1")
        m = self.base_model()
        m["actions"]["rescued"] = {"go": {
            "lost": {"lower": "1", "upper": "1"}}}
        self.assert_failure(m, "TERMINAL_HAS_ACTIONS", "actions.rescued")
        m = self.base_model()
        m["start"] = "nowhere"
        self.assert_failure(m, "UNKNOWN_STATE_REFERENCE", "start")

    def test_unknown_route_404(self):
        status, body = post(self.url("/nope"), {})
        self.assertEqual(status, 404)
        self.assertFalse(body["ok"])


if __name__ == "__main__":
    unittest.main()
