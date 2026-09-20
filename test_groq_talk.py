"""
Isolated test for talk_via_groq()
==================================
Step 1 of the Groq rollout plan -- prove the function works in isolation
before trusting it inside the full pipeline.

Usage
-----
Option A: rely on env var already set in your terminal session
    python test_groq_talk.py

Option B: hard-code the key just for this test run
    Set PASTE_KEY_HERE below to your actual key.

Pass criteria
-------------
  Prints a short natural spoken sentence (e.g. "Calculator's showing 94, boss.")
  Response time is under ~2 s (Groq is fast; >2 s suggests a network issue).
  Does NOT print None.

If it prints None
-----------------
  The line just above will tell you why:
    [groq] fell back to local -- GROQ_API_KEY is not set   -> fix step 0
    [groq] fell back to local -- 401 ...                   -> bad/wrong key
    [groq] fell back to local -- <model not found>         -> check GROQ_MODEL
    [groq] fell back to local -- <connection error>        -> firewall / network
"""
import os
import time

# -- Optional: hard-code your key here for a one-off test --------------------
# PASTE_KEY_HERE = "gsk_..."        # uncomment and fill in if you prefer
# os.environ["GROQ_API_KEY"] = PASTE_KEY_HERE
# ---------------------------------------------------------------------------

# Force the backend so talk_via_groq is actually tested even if
# ULTRON_TALK_BACKEND is not set in this shell session.
os.environ.setdefault("ULTRON_TALK_BACKEND", "groq")

from agent_server import talk_via_groq, GROQ_MODEL, GROQ_API_KEY

print("=" * 60)
print("Groq isolated talk test")
print(f"  Model   : {GROQ_MODEL}")
key_display = ("YES  (" + GROQ_API_KEY[:8] + "...)") if GROQ_API_KEY else "NO -- set GROQ_API_KEY first"
print(f"  Key set : {key_display}")
print("=" * 60)

messages = [
    {
        "role": "system",
        "content": "You are Ultron. Give one short spoken sentence confirming a tool result.",
    },
    {
        "role": "user",
        "content": (
            "Original request: open calculator and add 25+69\n\n"
            "Tool results:\n"
            "open_application: opened calc\n"
            "Calculator shows: 94"
        ),
    },
]

t0 = time.perf_counter()
result = talk_via_groq(messages)
elapsed = time.perf_counter() - t0

print(f"\nRESULT : {result}")
print(f"TIME   : {elapsed:.2f}s")

if result is None:
    print("\n[FAIL] talk_via_groq returned None -- read the [groq] line above for the reason.")
elif elapsed > 3:
    print(f"\n[WARN] Response took {elapsed:.2f}s -- network may be slow, but result is valid.")
else:
    print("\n[PASS] Groq returned a valid reply within expected time.")
