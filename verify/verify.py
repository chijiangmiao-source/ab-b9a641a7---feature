"""End-to-end verifier for the reachability-audit service.

Runs three stages and exits non-zero if any of them fails:

1. code tests   -- the unit/API test suite (exact rational solver,
                   certificates, validation failures)
2. build check  -- every service module byte-compiles and imports cleanly
3. API/HTTP smoke -- against a live service (BASE_URL, default
                   http://app:8080): health endpoint, exact fraction
                   probabilities (1/3, 2/3, certain rescue), canonical
                   lexicographic action on ties, zero rescue probability and
                   closed-loop identification, and locatable 400 failures
                   that must not produce success certificates.

The process exits after the run and prints VERIFY_EXIT_CODE so that
`docker compose up --exit-code-from verify` (or a shell) can report it.
"""

import json
import os
import py_compile
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = os.environ.get("BASE_URL", "http://app:8080").rstrip("/")
AUDIT_URL = BASE_URL + "/api/reachability-audit"
HEALTH_URL = BASE_URL + "/health"

_failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print("  [%s] %s%s" % (status, label, (" -- " + detail) if detail else ""))
    if not condition:
        _failures.append(label)


def stage_code_tests():
    print("== stage 1/3: code tests (unit + API suite) ==")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests",
         "-t", ".", "-v"],
        cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    check("unit/API test suite", proc.returncode == 0,
          "exit code %d" % proc.returncode)


def stage_build_check():
    print("== stage 2/3: build check (byte-compile + import) ==")
    ok = True
    for rel in ("app", "verify", "tests"):
        for dirpath, _dirnames, filenames in os.walk(os.path.join(ROOT, rel)):
            for name in sorted(filenames):
                if name.endswith(".py"):
                    path = os.path.join(dirpath, name)
                    try:
                        py_compile.compile(path, doraise=True)
                    except py_compile.PyCompileError as exc:
                        ok = False
                        print("  compile error: %s" % exc)
    check("byte-compile all modules", ok)
    proc = subprocess.run(
        [sys.executable, "-c",
         "import app.server, app.solver, verify.verify; print('imports ok')"],
        cwd=ROOT, capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write("  " + proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    check("import app.server / app.solver", proc.returncode == 0)


# --------------------------------------------------------------------------
# HTTP smoke helpers
# --------------------------------------------------------------------------

def http_post(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def wait_for_health(attempts=60, delay=1.0):
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(delay)
    return False


def stage_http_smoke():
    print("== stage 3/3: API/HTTP smoke against %s ==" % BASE_URL)
    check("health endpoint reachable", wait_for_health(), HEALTH_URL)

    # -- certain rescue: exact probability 1 --------------------------------
    status, body = http_post(AUDIT_URL, {
        "states": ["s0", "s1", "rescued", "lost"],
        "start": "s0",
        "rescuedStates": ["rescued"],
        "lostStates": ["lost"],
        "actions": {
            "s0": {"push": {"s1": "1/2", "rescued": "1/2"}},
            "s1": {"push": {"rescued": "1"}},
        },
    })
    check("certain-rescue model HTTP 200", status == 200, "got %s" % status)
    check("certain rescue probability is exactly 1",
          body.get("maxRescueProbability") == "1",
          repr(body.get("maxRescueProbability")))

    # -- 1/3 and 2/3 transitions: exact fractions in the answer -------------
    status, body = http_post(AUDIT_URL, {
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
    values = body.get("stateValues", {})
    check("thirds model HTTP 200", status == 200, "got %s" % status)
    check("start value is exact 2/3",
          body.get("maxRescueProbability") == "2/3",
          repr(body.get("maxRescueProbability")))
    check("state value is exact 1/3", values.get("s2") == "1/3",
          repr(values.get("s2")))

    # -- tied actions: canonical lexicographic choice, equal expectations ---
    status, body = http_post(AUDIT_URL, {
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
    chosen = body.get("optimalActions", {}).get("s0")
    check("tie model HTTP 200", status == 200, "got %s" % status)
    check("canonical action is lexicographically smallest",
          chosen == "alpha", repr(chosen))
    certs = body.get("certificates", [])
    entries = {}
    if certs:
        entries = {a["action"]: a for a in certs[0]["actions"]}
    tied = [entries.get("alpha", {}).get("expectedValue"),
            entries.get("zulu", {}).get("expectedValue")]
    check("tied actions show equal expected values",
          tied == ["1/2", "1/2"], repr(tied))
    check("dominated action excluded with visible expectation",
          entries.get("mike", {}).get("expectedValue") == "1/4"
          and entries.get("mike", {}).get("optimal") is False,
          repr(entries.get("mike")))

    # -- closed loop unreachable to terminals: zero rescue probability ------
    status, body = http_post(AUDIT_URL, {
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
    values = body.get("stateValues", {})
    check("closed-loop model HTTP 200", status == 200, "got %s" % status)
    check("forced-into-loop start has zero rescue probability",
          body.get("maxRescueProbability") == "0",
          repr(body.get("maxRescueProbability")))
    check("loop states have zero rescue probability",
          values.get("loop1") == "0" and values.get("loop2") == "0",
          repr(values))
    check("closed loop identified as non-terminating",
          body.get("nonTerminatingStates") == ["loop1", "loop2", "s0"],
          repr(body.get("nonTerminatingStates")))

    # -- invalid models: locatable 400, no success certificate --------------
    bad_models = [
        ("unknown transition target",
         {"states": ["s0", "rescued"], "start": "s0",
          "rescuedStates": ["rescued"], "lostStates": [],
          "actions": {"s0": {"go": {"s9": "1"}}}},
         "UNKNOWN_STATE_REFERENCE"),
        ("non-terminal without actions",
         {"states": ["s0", "s1", "rescued"], "start": "s0",
          "rescuedStates": ["rescued"], "lostStates": [],
          "actions": {"s0": {"go": {"rescued": "1"}}}},
         "NONTERMINAL_MISSING_ACTIONS"),
        ("terminal carrying actions",
         {"states": ["s0", "rescued"], "start": "s0",
          "rescuedStates": ["rescued"], "lostStates": [],
          "actions": {"s0": {"go": {"rescued": "1"}},
                      "rescued": {"stay": {"rescued": "1"}}}},
         "TERMINAL_HAS_ACTIONS"),
        ("probabilities not summing to one",
         {"states": ["s0", "rescued", "lost"], "start": "s0",
          "rescuedStates": ["rescued"], "lostStates": ["lost"],
          "actions": {"s0": {"go": {"rescued": "1/3", "lost": "1/3"}}}},
         "PROBABILITY_SUM_INVALID"),
    ]
    for label, payload, code in bad_models:
        status, body = http_post(AUDIT_URL, payload)
        error = body.get("error", {})
        check("invalid model rejected: %s" % label,
              status == 400
              and body.get("ok") is False
              and error.get("code") == code
              and error.get("path")
              and "certificates" not in body
              and "maxRescueProbability" not in body,
              "status=%s body=%s" % (status, json.dumps(body)[:200]))


def main():
    print("reachability-audit verifier; BASE_URL=%s" % BASE_URL)
    stage_code_tests()
    stage_build_check()
    stage_http_smoke()

    print("=" * 60)
    if _failures:
        print("VERIFY FAILED (%d check(s)):" % len(_failures))
        for label in _failures:
            print("  - %s" % label)
        exit_code = 1
    else:
        print("VERIFY OK: all checks passed")
        exit_code = 0
    print("VERIFY_EXIT_CODE=%d" % exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
