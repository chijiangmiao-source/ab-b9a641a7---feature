"""HTTP API for the deep-sea buoy reachability audit service.

Endpoints
---------
GET  /health                          liveness probe used by Docker/Compose
POST /api/reachability-audit          exact maximum-rescue-probability audit
POST /api/robust-reachability-audit   robust (interval-MDP) worst-case audit

The audit endpoint accepts one JSON model: unique states, a start state,
rescued and lost terminal states, and for every controllable (non-terminal)
state its unique actions, each with a transition distribution given as
reduced fractions ("p/q").  It replies with the exact maximum rescue
probability, the canonical optimal action of every controllable state and an
itemised certificate per state so the caller can recompute why each action
was selected or excluded.  Invalid models (unknown references, non-terminal
states without actions, terminals carrying actions, transition probabilities
that do not sum to exactly one, ...) are rejected with a locatable error and
never produce a success certificate.

The robust audit takes the same state/action structure, but each possible
successor of an action carries a closed probability interval "[lo, hi]"
(reduced fractions).  After every action the link disturbance may afresh pick
any distribution over the successors that respects every interval and sums to
exactly one; the controller maximises and the disturbance minimises the
infinite-horizon rescue probability.  Every reply itemises, per action, the
canonical worst-case (minimising) distribution and its exact expectation.
Models whose intervals admit no distribution summing to one are rejected with
a locatable error.

Only the Python standard library is used; all arithmetic is exact rational
arithmetic via fractions.Fraction.
"""

import json
import os
import re
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .robust_solver import solve_robust_reachability
from .solver import solve_reachability

AUDIT_PATH = "/api/reachability-audit"
ROBUST_AUDIT_PATH = "/api/robust-reachability-audit"
HEALTH_PATH = "/health"

_FRACTION_RE = re.compile(r"^([+-]?\d+)(?:/(\d+))?$")


class ClientError(Exception):
    """A locatable, client-side model/validation failure (HTTP 400)."""

    def __init__(self, code, message, path):
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


def _loads_strict(text):
    """Parse JSON, rejecting duplicate object keys (states/actions must be
    uniquely identified)."""
    def hook(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ClientError(
                    "DUPLICATE_KEY",
                    "duplicate object key %r; identifiers must be unique" % key,
                    key,
                )
            obj[key] = value
        return obj
    try:
        return json.loads(text, object_pairs_hook=hook)
    except json.JSONDecodeError as exc:
        raise ClientError("INVALID_JSON",
                          "request body is not valid JSON: %s" % exc,
                          None) from exc


def _require(condition, code, message, path):
    if not condition:
        raise ClientError(code, message, path)


def _parse_probability(value, path):
    if isinstance(value, bool):
        raise ClientError("INVALID_FRACTION",
                          "probability must be a reduced fraction string "
                          "'p/q', not a boolean", path)
    if isinstance(value, int):
        frac = Fraction(value)
    elif isinstance(value, str):
        match = _FRACTION_RE.match(value.strip())
        if not match:
            raise ClientError(
                "INVALID_FRACTION",
                "probability %r is not a fraction of the form 'p' or 'p/q'"
                % value, path)
        numerator = int(match.group(1))
        denominator = int(match.group(2)) if match.group(2) else 1
        if denominator == 0:
            raise ClientError("INVALID_FRACTION",
                              "probability %r has a zero denominator" % value,
                              path)
        frac = Fraction(numerator, denominator)
    else:
        raise ClientError(
            "INVALID_FRACTION",
            "probability must be a reduced fraction string 'p/q'; "
            "floating-point values are not accepted", path)
    if not (Fraction(0) <= frac <= Fraction(1)):
        raise ClientError("INVALID_PROBABILITY",
                          "probability %s is outside [0, 1]" % frac, path)
    return frac


def _fmt(frac):
    """Render an exact Fraction: '2/3' for rationals, '1'/'0' for integers."""
    return str(frac)


def _parse_common_structure(payload, transition_kind):
    """Validate and return the state/terminal/action shell shared by both
    audits: unique states, start, disjoint terminal sets and the placement of
    action blocks (only non-terminal states may carry a non-empty block).

    The per-transition bodies are left to the caller, which supplies
    ``transition_kind`` (a human label used in error messages).
    """
    _require(isinstance(payload, dict), "INVALID_SCHEMA",
             "request body must be a JSON object", None)

    for field in ("states", "start", "rescuedStates", "lostStates", "actions"):
        _require(field in payload, "INVALID_SCHEMA",
                 "missing required field %r" % field, field)

    # ---- unique states -------------------------------------------------
    states = payload["states"]
    _require(isinstance(states, list) and states, "INVALID_SCHEMA",
             "'states' must be a non-empty array of identifiers", "states")
    seen = set()
    for i, state in enumerate(states):
        path = "states[%d]" % i
        _require(isinstance(state, str) and state, "INVALID_SCHEMA",
                 "state identifiers must be non-empty strings", path)
        _require(state not in seen, "DUPLICATE_STATE",
                 "duplicate state identifier %r" % state, path)
        seen.add(state)
    state_set = seen

    # ---- start state ----------------------------------------------------
    start = payload["start"]
    _require(isinstance(start, str), "INVALID_SCHEMA",
             "'start' must be a state identifier string", "start")
    _require(start in state_set, "UNKNOWN_STATE_REFERENCE",
             "start state %r is not declared in 'states'" % start, "start")

    # ---- terminal states -------------------------------------------------
    def parse_terminals(field):
        raw = payload[field]
        _require(isinstance(raw, list), "INVALID_SCHEMA",
                 "%r must be an array of state identifiers" % field, field)
        result = []
        local_seen = set()
        for i, state in enumerate(raw):
            path = "%s[%d]" % (field, i)
            _require(isinstance(state, str) and state, "INVALID_SCHEMA",
                     "terminal identifiers must be non-empty strings", path)
            _require(state in state_set, "UNKNOWN_STATE_REFERENCE",
                     "terminal state %r is not declared in 'states'" % state,
                     path)
            _require(state not in local_seen, "DUPLICATE_STATE",
                     "duplicate terminal identifier %r" % state, path)
            local_seen.add(state)
            result.append(state)
        return result

    rescued = parse_terminals("rescuedStates")
    lost = parse_terminals("lostStates")
    overlap = set(rescued) & set(lost)
    _require(not overlap, "TERMINAL_CONFLICT",
             "states %s are declared both rescued and lost"
             % sorted(overlap), "rescuedStates")
    terminals = set(rescued) | set(lost)

    # ---- action block shell ---------------------------------------------
    raw_actions = payload["actions"]
    _require(isinstance(raw_actions, dict), "INVALID_SCHEMA",
             "'actions' must be an object mapping state -> action -> "
             "transition " + transition_kind, "actions")
    for state, state_actions in raw_actions.items():
        path = "actions.%s" % state
        _require(state in state_set, "UNKNOWN_STATE_REFERENCE",
                 "actions declared for unknown state %r" % state, path)
        _require(state not in terminals, "TERMINAL_HAS_ACTIONS",
                 "terminal state %r must not carry actions" % state, path)
        _require(isinstance(state_actions, dict), "INVALID_SCHEMA",
                 "actions of state %r must be an object" % state, path)
        _require(bool(state_actions), "NONTERMINAL_MISSING_ACTIONS",
                 "controllable state %r has no actions" % state, path)
        for action_id, dist in state_actions.items():
            apath = "%s.%s" % (path, action_id)
            _require(isinstance(action_id, str) and action_id,
                     "INVALID_SCHEMA", "action identifiers must be non-empty "
                     "strings", apath)
            _require(isinstance(dist, dict), "INVALID_SCHEMA",
                     "transition %s of action %r must be an object mapping "
                     "state -> %s"
                     % (transition_kind, action_id, transition_kind), apath)

    for state in states:
        if state not in terminals:
            _require(state in raw_actions, "NONTERMINAL_MISSING_ACTIONS",
                     "controllable state %r has no actions" % state,
                     "actions.%s" % state)

    return states, start, rescued, lost, terminals, raw_actions


def parse_robust_model(payload):
    """Validate an interval-model request into exact rational intervals.

    Each transition body is ``{"lower": "p/q", "upper": "p/q"}`` (bare
    integers accepted as well).  Per action the closed boxes must contain at
    least one distribution summing to exactly one, i.e.
    ``sum lower <= 1 <= sum upper`` with ``lower <= upper``; otherwise the
    request is rejected with a path locating the offending action/target.
    """
    states, start, rescued, lost, terminals, raw_actions = \
        _parse_common_structure(payload, "interval object")

    state_set = set(states)
    actions = {}
    for state, state_actions in raw_actions.items():
        parsed_state_actions = {}
        for action_id, dist in state_actions.items():
            apath = "actions.%s.%s" % (state, action_id)
            intervals = {}
            lower_sum = Fraction(0)
            upper_sum = Fraction(0)
            for target, box in dist.items():
                tpath = "%s.%s" % (apath, target)
                _require(target in state_set, "UNKNOWN_STATE_REFERENCE",
                         "transition target %r is not declared in 'states'"
                         % target, tpath)
                _require(isinstance(box, dict)
                         and not isinstance(box, bool), "INTERVAL_INVALID",
                         "transition to %r must be an interval object "
                         "{'lower': 'p/q', 'upper': 'p/q'}" % target, tpath)
                for endpoint in ("lower", "upper"):
                    _require(endpoint in box, "INTERVAL_INVALID",
                             "interval to %r is missing its %r bound"
                             % (target, endpoint), tpath)
                lo = _parse_probability(box["lower"],
                                        "%s.lower" % tpath)
                hi = _parse_probability(box["upper"],
                                        "%s.upper" % tpath)
                _require(lo <= hi, "INTERVAL_INVALID",
                         "interval to %r is inverted: lower %s > upper %s"
                         % (target, lo, hi), tpath)
                intervals[target] = (lo, hi)
                lower_sum += lo
                upper_sum += hi
            _require(lower_sum <= 1, "PROBABILITY_SUM_INVALID",
                     "intervals of action %r in state %r force a total mass of "
                     "at least %s, which exceeds 1"
                     % (action_id, state, lower_sum), apath)
            _require(upper_sum >= 1, "PROBABILITY_SUM_INVALID",
                     "intervals of action %r in state %r allow a total mass of "
                     "at most %s, which is below 1"
                     % (action_id, state, upper_sum), apath)
            parsed_state_actions[action_id] = intervals
        actions[state] = parsed_state_actions

    return {
        "states": states,
        "start": start,
        "rescued": rescued,
        "lost": lost,
        "actions": actions,
    }


def parse_model(payload):
    """Validate the request payload and return the exact rational model.

    Raises :class:`ClientError` with a locatable path on any failure.
    """
    states, start, rescued, lost, terminals, raw_actions = \
        _parse_common_structure(payload, "fraction")

    state_set = set(states)
    actions = {}
    for state, state_actions in raw_actions.items():
        parsed_state_actions = {}
        for action_id, dist in state_actions.items():
            apath = "actions.%s.%s" % (state, action_id)
            parsed_dist = {}
            total = Fraction(0)
            for target, raw_prob in dist.items():
                tpath = "%s.%s" % (apath, target)
                _require(target in state_set, "UNKNOWN_STATE_REFERENCE",
                         "transition target %r is not declared in 'states'"
                         % target, tpath)
                prob = _parse_probability(raw_prob, tpath)
                parsed_dist[target] = prob
                total += prob
            _require(total == 1, "PROBABILITY_SUM_INVALID",
                     "transition probabilities of action %r in state %r sum "
                     "to %s, not exactly 1" % (action_id, state, total), apath)
            parsed_state_actions[action_id] = parsed_dist
        actions[state] = parsed_state_actions

    return {
        "states": states,
        "start": start,
        "rescued": rescued,
        "lost": lost,
        "actions": actions,
    }


def build_success_payload(result):
    certificates = []
    for cert in result["certificates"]:
        certificates.append({
            "state": cert["state"],
            "value": _fmt(cert["value"]),
            "selectedAction": cert["selectedAction"],
            "actions": [{
                "action": entry["action"],
                "expectedValue": _fmt(entry["expectedValue"]),
                "optimal": entry["optimal"],
                "selected": entry["selected"],
            } for entry in cert["actions"]],
        })
    return {
        "ok": True,
        "start": result["start"],
        "maxRescueProbability": _fmt(result["maxRescueProbability"]),
        "stateValues": {s: _fmt(v) for s, v in result["stateValues"].items()},
        "optimalActions": result["optimalActions"],
        "certificates": certificates,
        "nonTerminatingStates": result["nonTerminatingStates"],
    }


def build_robust_success_payload(result):
    certificates = []
    for cert in result["certificates"]:
        actions = []
        for entry in cert["actions"]:
            targets = entry["targets"]
            actions.append({
                "action": entry["action"],
                "expectedValue": _fmt(entry["expectedValue"]),
                "optimal": entry["optimal"],
                "selected": entry["selected"],
                "worstDistribution": {
                    t: {
                        "lower": _fmt(entry["lower"][t]),
                        "upper": _fmt(entry["upper"][t]),
                        "probability": _fmt(entry["worstDistribution"][t]),
                    } for t in targets
                },
            })
        certificates.append({
            "state": cert["state"],
            "value": _fmt(cert["value"]),
            "selectedAction": cert["selectedAction"],
            "actions": actions,
        })
    return {
        "ok": True,
        "start": result["start"],
        "maxRescueProbability": _fmt(result["maxRescueProbability"]),
        "stateValues": {s: _fmt(v) for s, v in result["stateValues"].items()},
        "optimalActions": result["optimalActions"],
        "certificates": certificates,
        "nonTerminatingStates": result["nonTerminatingStates"],
    }


class AuditHandler(BaseHTTPRequestHandler):
    server_version = "ReachabilityAudit/1.0"
    protocol_version = "HTTP/1.1"

    # -- helpers ----------------------------------------------------------
    def _send_json(self, status, obj):
        body = json.dumps(obj, indent=2, sort_keys=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status, code, message, path=None):
        self._send_json(status, {
            "ok": False,
            "error": {"code": code, "message": message, "path": path},
        })

    def log_message(self, fmt, *args):  # keep access logs on one line
        import sys
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- routes -------------------------------------------------------------
    def do_GET(self):
        if self.path == HEALTH_PATH:
            self._send_json(200, {"ok": True, "status": "healthy"})
        elif self.path == "/":
            self._send_json(200, {
                "ok": True,
                "service": "reachability-audit",
                "endpoints": {"audit": "POST " + AUDIT_PATH,
                              "robustAudit": "POST " + ROBUST_AUDIT_PATH,
                              "health": "GET " + HEALTH_PATH},
            })
        else:
            self._send_error(404, "NOT_FOUND", "no such route: %s" % self.path)

    def do_POST(self):
        if self.path not in (AUDIT_PATH, ROBUST_AUDIT_PATH):
            self._send_error(404, "NOT_FOUND", "no such route: %s" % self.path)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_error(400, "INVALID_REQUEST", "bad Content-Length")
            return
        try:
            raw = self.rfile.read(length)
            payload = _loads_strict(raw.decode("utf-8"))
            if self.path == AUDIT_PATH:
                model = parse_model(payload)
            else:
                model = parse_robust_model(payload)
        except ClientError as exc:
            self._send_error(400, exc.code, exc.message, exc.path)
            return
        except (UnicodeDecodeError, ValueError) as exc:
            self._send_error(400, "INVALID_REQUEST", str(exc))
            return

        if self.path == AUDIT_PATH:
            result = solve_reachability(model["states"], model["start"],
                                        model["rescued"], model["lost"],
                                        model["actions"])
            self._send_json(200, build_success_payload(result))
        else:
            result = solve_robust_reachability(
                model["states"], model["start"], model["rescued"],
                model["lost"], model["actions"])
            self._send_json(200, build_robust_success_payload(result))


def main():
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), AuditHandler)
    print("reachability-audit listening on 0.0.0.0:%d" % port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
