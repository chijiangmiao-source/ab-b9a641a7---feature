# Reachability Audit Service (深海浮标可达性审计)

Exact maximum-rescue-probability audits for a deep-sea buoy failure model.
Given the unique states of the buoy, a start state, rescued/lost terminal
states and, for every controllable state, its unique actions with
reduced-fraction transition distributions, the service computes:

* the **maximum rescue probability** of the start state,
* the **canonical optimal action** of every controllable state (among all
  optimal actions, the smallest identifier), and
* an **itemised certificate** per state: the state value and the expected
  value of every action, so the caller can recompute why each action was
  selected or excluded.

## How it solves (exactness guarantees)

The maximal reachability probability vector of a finite MDP is the least
fixed point of the Bellman optimality operator, i.e. the unique
componentwise-minimal solution of

```
v(s) >= sum_{s'} p(s,a,s') * v(s')   for every controllable s and action a
v(r) >= 1                            for every rescued terminal r
v(s) >= 0
```

The service minimises `sum_s v(s)` over this polytope with a **two-phase
simplex method in arbitrary-precision rational arithmetic**
(`fractions.Fraction`, Bland's pivoting rule). It never uses floating point,
bounded random simulation, policy enumeration or truncated iteration; the
simplex method terminates after finitely many exact rational pivots with the
provably optimal solution. States trapped in closed loops that can never
reach any terminal are identified by an exact support-graph reachability
pass and reported in `nonTerminatingStates` (their rescue probability is
exactly `0`).

## API

### `POST /api/reachability-audit`

Request:

```json
{
  "states": ["s0", "s1", "rescued", "lost"],
  "start": "s0",
  "rescuedStates": ["rescued"],
  "lostStates": ["lost"],
  "actions": {
    "s0": {"commit": {"s1": "2/3", "lost": "1/3"}},
    "s1": {"retry": {"rescued": "1/2", "s1": "1/2"}}
  }
}
```

* `states` — unique, non-empty state identifiers.
* `start` — must be one of `states`.
* `rescuedStates` / `lostStates` — disjoint subsets of `states` (terminals).
* `actions` — exactly the non-terminal states, each with uniquely named
  actions; every action maps target states to reduced fractions `"p/q"`
  whose sum must be **exactly** `1`.

Success (`200`):

```json
{
  "ok": true,
  "start": "s0",
  "maxRescueProbability": "2/3",
  "stateValues": {"s0": "2/3", "s1": "1", "rescued": "1", "lost": "0"},
  "optimalActions": {"s0": "commit", "s1": "retry"},
  "certificates": [
    {"state": "s0", "value": "2/3", "selectedAction": "commit",
     "actions": [{"action": "commit", "expectedValue": "2/3",
                  "optimal": true, "selected": true}]}
  ],
  "nonTerminatingStates": []
}
```

All probabilities are exact reduced fractions (`"2/3"`, `"1"`, `"0"`).
`optimalActions` picks, among all actions whose expected value equals the
state value, the lexicographically smallest identifier; the certificate
lists every action's expected value with `optimal`/`selected` flags.

Failure (`400`, locatable, never a success certificate):

```json
{"ok": false,
 "error": {"code": "PROBABILITY_SUM_INVALID",
           "message": "transition probabilities of action 'go' in state 's0' sum to 2/3, not exactly 1",
           "path": "actions.s0.go"}}
```

Error codes include `UNKNOWN_STATE_REFERENCE`,
`NONTERMINAL_MISSING_ACTIONS`, `TERMINAL_HAS_ACTIONS`,
`PROBABILITY_SUM_INVALID`, `INVALID_FRACTION`, `INVALID_PROBABILITY`,
`DUPLICATE_STATE`, `DUPLICATE_KEY`, `TERMINAL_CONFLICT`, `INVALID_SCHEMA`,
`INVALID_JSON`.

### `GET /health`

Liveness probe: `200 {"ok": true, "status": "healthy"}`.

## Run with Docker Compose

```sh
docker compose up --build app                 # http://localhost:8080
HOST_PORT=9000 docker compose up --build app  # host port is configurable
curl -s localhost:9000/health                 # health check works
```

## Verify (tests + build check + HTTP smoke, then exits)

The `verify` service waits for `app` to be healthy, then runs the unit/API
test suite, a byte-compile/import build check, and live HTTP smoke checks
(exact `1/3`/`2/3` fractions, certain rescue, canonical lexicographic action
on ties with equal expected values, zero rescue probability for the closed
loop and the start state forced into it, and locatable `400` failures). It
then exits and reports its exit code:

```sh
docker compose up --build --exit-code-from verify
echo "verify exit code: $?"
```

## Local development (no Docker)

```sh
python3 -m unittest discover -s tests -t . -v   # test suite
PORT=8080 python3 -m app.server                 # serve
BASE_URL=http://127.0.0.1:8080 python3 -m verify.verify
```

## Layout

```
app/solver.py    exact rational LP (two-phase simplex) + model solver
app/server.py    HTTP API, request validation, exact fraction rendering
tests/           unit + API tests (unittest, stdlib only)
verify/verify.py compose verify service: tests, build check, HTTP smoke
Dockerfile       python:3.12-slim image with HEALTHCHECK
docker-compose.yml  app (configurable HOST_PORT) + one-shot verify
```
