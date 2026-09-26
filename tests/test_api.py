"""HTTP/API tests: exact fraction responses, certificates and every
locatable validation failure (which must never yield a success certificate)."""

import json
import threading
import unittest
import urllib.error
import urllib.request

from app.server import AuditHandler
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


class ApiTest(unittest.TestCase):
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
    def assert_failure(self, payload, code, path_prefix=None, raw=None):
        status, body = post(self.url("/api/reachability-audit"), payload,
                            raw=raw)
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


if __name__ == "__main__":
    unittest.main()
