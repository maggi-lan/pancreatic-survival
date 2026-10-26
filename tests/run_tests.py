"""Minimal runner so tests work without pytest:  python tests/run_tests.py"""
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import test_core
fails = 0
for name in sorted(n for n in dir(test_core) if n.startswith("test_")):
    try:
        getattr(test_core, name)(); print("PASS", name)
    except Exception:
        fails += 1; print("FAIL", name); traceback.print_exc()
sys.exit(1 if fails else 0)
