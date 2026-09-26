"""HTTP API for the deep-sea buoy reachability audit service.

Endpoints
---------
GET  /health                           liveness probe used by Docker/Compose
POST /api/reachability-audit           exact maximum-rescue-probability audit
POST /api/robust-reachability-audit    exact robust (interval-valued) audit

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

The robust audit endpoint accepts the same model shape, except that every
transition carries an interval {"lower": "p/q", "upper": "p/q"} of reduced
fractions.  Every action's intervals must admit a distribution summing to
exactly one (sum of lower bounds <= 1 <= sum of upper bounds), otherwise the
model is rejected with an error located at the offending action or target
state.  The reply is the exact controller-maximising / perturbation-
minimising infinite-horizon rescue probability: after every action the
perturbation may re-select any distribution inside the intervals.  Each
certificate action additionally lists the canonical worst-case distribution
attaining its expected value.  For point intervals (lower == upper
everywhere) the robust reply agrees with the plain audit item by item.

Only the Python standard library is used; all arithmetic is exact rational
arithmetic via fractions.Fraction.
"""

import json
import os
import re
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .solver import solve_reachability, solve_robust_reachability

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


def _parse_probability(value, path, what="probability"):
    if isinstance(value, bool):
        raise ClientError("INVALID_FRACTION",
                          "%s must be a reduced fraction string "
                          "'p/q', not a boolean" % what, path)
    if isinstance(value, int):
        frac = Fraction(value)
    elif isinstance(value, str):
        match = _FRACTION_RE.match(value.strip())
        if not match:
            raise ClientError(
                "INVALID_FRACTION",
                "%s %r is not a fraction of the form 'p' or 'p/q'"
                % (what, value), path)
        numerator = int(match.group(1))
        denominator = int(match.group(2)) if match.group(2) else 1
        if denominator == 0:
            raise ClientError("INVALID_FRACTION",
                              "%s %r has a zero denominator" % (what, value),
                              path)
        frac = Fraction(numerator, denominator)
    else:
        raise ClientError(
            "INVALID_FRACTION",
            "%s must be a reduced fraction string 'p/q'; "
            "floating-point values are not accepted" % what, path)
    if not (Fraction(0) <= frac <= Fraction(1)):
        raise ClientError("INVALID_PROBABILITY",
                          "%s %s is outside [0, 1]" % (what, frac), path)
    return frac


def _parse_common(payload):
    """Validate everything the two audit variants share: unique states, the
    start state, disjoint rescued/lost terminals and the actions envelope.

    Returns ``(states, state_set, start, rescued, lost, terminals,
    raw_actions)``.  Raises :class:`ClientError` with a locatable path on any
    failure.
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

    # ---- actions envelope --------------------------------------------------
    raw_actions = payload["actions"]
    _require(isinstance(raw_actions, dict), "INVALID_SCHEMA",
             "'actions' must be an object mapping state -> action -> "
             "transition distribution", "actions")

    return states, state_set, start, rescued, lost, terminals, raw_actions


def _parse_actions(raw_actions, state_set, terminals, parse_dist):
    """Validate the per-state/per-action structure and parse every action's
    transitions with ``parse_dist(state, action_id, dist, apath)``."""
    actions = {}
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
        parsed_state_actions = {}
        for action_id, dist in state_actions.items():
            apath = "%s.%s" % (path, action_id)
            _require(isinstance(action_id, str) and action_id,
                     "INVALID_SCHEMA", "action identifiers must be non-empty "
                     "strings", apath)
            parsed_state_actions[action_id] = parse_dist(
                state, action_id, dist, apath)
        actions[state] = parsed_state_actions
    return actions


def _require_action_coverage(states, terminals, actions):
    for state in states:
        if state not in terminals:
            _require(state in actions, "NONTERMINAL_MISSING_ACTIONS",
                     "controllable state %r has no actions" % state,
                     "actions.%s" % state)


def parse_model(payload):
    """Validate the request payload and return the exact rational model.

    Raises :class:`ClientError` with a locatable path on any failure.
    """
    (states, state_set, start, rescued, lost, terminals,
     raw_actions) = _parse_common(payload)

    def parse_dist(state, action_id, dist, apath):
        _require(isinstance(dist, dict), "INVALID_SCHEMA",
                 "transition distribution of action %r must be an object "
                 "mapping state -> fraction" % action_id, apath)
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
        return parsed_dist

    actions = _parse_actions(raw_actions, state_set, terminals, parse_dist)
    _require_action_coverage(states, terminals, actions)

    return {
        "states": states,
        "start": start,
        "rescued": rescued,
        "lost": lost,
        "actions": actions,
    }


def parse_interval_model(payload):
    """Validate a robust-audit payload and return the interval model.

    Same structure as :func:`parse_model`, except every transition maps its
    target to ``{"lower": "p/q", "upper": "p/q"}``.  Each action's intervals
    must admit a distribution summing to exactly one: lower bounds summing
    to more than 1 or upper bounds summing to less than 1 are rejected with
    ``INTERVAL_SUM_INVALID`` located at the action; a lower bound above its
    upper bound is rejected with ``INTERVAL_EMPTY`` located at the target.
    """
    (states, state_set, start, rescued, lost, terminals,
     raw_actions) = _parse_common(payload)

    def parse_dist(state, action_id, dist, apath):
        _require(isinstance(dist, dict), "INVALID_SCHEMA",
                 "transition intervals of action %r must be an object "
                 "mapping state -> {'lower': f, 'upper': f}" % action_id,
                 apath)
        parsed_dist = {}
        lower_total = Fraction(0)
        upper_total = Fraction(0)
        for target, raw_bounds in dist.items():
            tpath = "%s.%s" % (apath, target)
            _require(target in state_set, "UNKNOWN_STATE_REFERENCE",
                     "transition target %r is not declared in 'states'"
                     % target, tpath)
            _require(isinstance(raw_bounds, dict), "INVALID_SCHEMA",
                     "interval of target %r must be an object with 'lower' "
                     "and 'upper' bounds" % target, tpath)
            unexpected = sorted(set(raw_bounds) - {"lower", "upper"})
            _require(not unexpected, "INVALID_SCHEMA",
                     "interval of target %r has unexpected keys %s"
                     % (target, unexpected), tpath)
            for key in ("lower", "upper"):
                _require(key in raw_bounds, "INVALID_SCHEMA",
                         "interval of target %r is missing the %r bound"
                         % (target, key), "%s.%s" % (tpath, key))
            lower = _parse_probability(raw_bounds["lower"],
                                       "%s.lower" % tpath,
                                       "interval lower bound")
            upper = _parse_probability(raw_bounds["upper"],
                                       "%s.upper" % tpath,
                                       "interval upper bound")
            _require(lower <= upper, "INTERVAL_EMPTY",
                     "lower bound %s exceeds upper bound %s for target %r"
                     % (lower, upper, target), tpath)
            parsed_dist[target] = (lower, upper)
            lower_total += lower
            upper_total += upper
        _require(lower_total <= 1, "INTERVAL_SUM_INVALID",
                 "lower bounds of action %r in state %r sum to %s > 1; no "
                 "distribution within the intervals sums to exactly 1"
                 % (action_id, state, lower_total), apath)
        _require(upper_total >= 1, "INTERVAL_SUM_INVALID",
                 "upper bounds of action %r in state %r sum to %s < 1; no "
                 "distribution within the intervals sums to exactly 1"
                 % (action_id, state, upper_total), apath)
        return parsed_dist

    actions = _parse_actions(raw_actions, state_set, terminals, parse_dist)
    _require_action_coverage(states, terminals, actions)

    return {
        "states": states,
        "start": start,
        "rescued": rescued,
        "lost": lost,
        "actions": actions,
    }


def _fmt(frac):
    """Render an exact Fraction: '2/3' for rationals, '1'/'0' for integers."""
    return str(frac)


def build_success_payload(result):
    certificates = []
    for cert in result["certificates"]:
        entries = []
        for entry in cert["actions"]:
            item = {
                "action": entry["action"],
                "expectedValue": _fmt(entry["expectedValue"]),
            }
            if "worstDistribution" in entry:
                # robust audits: the canonical worst-case distribution that
                # attains this action's expected value
                item["worstDistribution"] = {
                    target: _fmt(prob)
                    for target, prob in sorted(
                        entry["worstDistribution"].items())
                }
            item["optimal"] = entry["optimal"]
            item["selected"] = entry["selected"]
            entries.append(item)
        certificates.append({
            "state": cert["state"],
            "value": _fmt(cert["value"]),
            "selectedAction": cert["selectedAction"],
            "actions": entries,
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
        routes = {
            AUDIT_PATH: (parse_model, solve_reachability),
            ROBUST_AUDIT_PATH: (parse_interval_model,
                                solve_robust_reachability),
        }
        route = routes.get(self.path)
        if route is None:
            self._send_error(404, "NOT_FOUND", "no such route: %s" % self.path)
            return
        parse, solve = route
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_error(400, "INVALID_REQUEST", "bad Content-Length")
            return
        try:
            raw = self.rfile.read(length)
            payload = _loads_strict(raw.decode("utf-8"))
            model = parse(payload)
        except ClientError as exc:
            self._send_error(400, exc.code, exc.message, exc.path)
            return
        except (UnicodeDecodeError, ValueError) as exc:
            self._send_error(400, "INVALID_REQUEST", str(exc))
            return

        result = solve(model["states"], model["start"], model["rescued"],
                       model["lost"], model["actions"])
        self._send_json(200, build_success_payload(result))


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
