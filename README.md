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

A second, **robust** audit accepts interval-valued transition probabilities
(lower/upper bounds as reduced fractions) and computes the exact
**controller-maximising / perturbation-minimising** rescue probability: after
every action the link perturbation may re-select any distribution inside the
intervals. Each certificate action additionally lists the **canonical
worst-case distribution** attaining its expected value.

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

### Robust (interval-valued) models

For the robust audit, every transition carries an interval
`{"lower": "p/q", "upper": "p/q"}` and the perturbation may pick **any**
distribution inside each action's interval polytope — adversarially, and
re-chosen on every visit. The guaranteed rescue probability is the least
fixed point of the max–min Bellman operator

```
v(s) = max_a  min_{p in Delta_a}  sum_{s'} p(s') * v(s')
```

computed **exactly** by constraint generation over the same rational
simplex: the polytope `v(s) >= p . v` for every vertex `p` of every
`Delta_a` has the least fixed point as its componentwise-minimal point, and
violated vertex constraints are found exactly by a greedy worst-vertex
minimisation (all targets at their lower bounds, the remaining mass poured
onto the lowest-value targets, ties broken canonically by target
identifier). No interval midpoints, no floating point, no random replay and
no truncated finite-round iteration are involved. For point intervals
(`lower == upper` everywhere) the robust reply agrees with the plain audit
item by item.

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

### `POST /api/robust-reachability-audit`

Same model shape as the plain audit, except that every transition maps its
target to an interval of reduced fractions:

```json
{
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
                    "lost": {"lower": "3/4", "upper": "3/4"}}}
  }
}
```

* every action's intervals must admit a distribution summing to **exactly**
  `1` (`sum(lower) <= 1 <= sum(upper)`), otherwise the model is rejected:
  `INTERVAL_SUM_INVALID` located at the action, or `INTERVAL_EMPTY` located
  at the target state when `lower > upper`.

Success (`200`) — the same fields as the plain audit, plus a canonical
`worstDistribution` per certificate action (the worst-case distribution
attaining its expected value, so the caller can recompute why each action
was selected or excluded):

```json
{
  "ok": true,
  "start": "s0",
  "maxRescueProbability": "3/8",
  "stateValues": {"s0": "3/8", "s1": "3/4", "s2": "1/4", "rescued": "1", "lost": "0"},
  "optimalActions": {"s0": "commit", "s1": "push", "s2": "call"},
  "certificates": [
    {"state": "s0", "value": "3/8", "selectedAction": "commit",
     "actions": [{"action": "commit", "expectedValue": "3/8",
                  "worstDistribution": {"s1": "1/4", "s2": "3/4"},
                  "optimal": true, "selected": true}]}
  ],
  "nonTerminatingStates": []
}
```

Here the free `1/2` of probability mass in `commit` can shift between the
two successors; the perturbation pours it onto the worse one, and the
guaranteed rescue probability is the exact lower bound `1/4*3/4 + 3/4*1/4 =
3/8`. Closed loops keep zero rescue probability, and models whose interval
sums admit no distribution are rejected with a locatable `400` and never
produce a success certificate. For point intervals the reply agrees with
`POST /api/reachability-audit` item by item.

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
loop and the start state forced into it, locatable `400` failures, and the
robust interval audit: item-by-item agreement with the plain audit on
degenerate point intervals, exact lower bounds for shiftable probability
mass, zero probability for interval closed loops, and rejections of
infeasible interval sums). It then exits and reports its exit code:

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
app/solver.py    exact rational LP (two-phase simplex) + model solvers
                 (plain and robust/interval, the latter by exact constraint
                 generation over interval-polytope vertices)
app/server.py    HTTP API, request validation, exact fraction rendering
tests/           unit + API tests (unittest, stdlib only)
verify/verify.py compose verify service: tests, build check, HTTP smoke
Dockerfile       python:3.12-slim image with HEALTHCHECK
docker-compose.yml  app (configurable HOST_PORT) + one-shot verify
```
