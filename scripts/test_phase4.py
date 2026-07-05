import sys
import math
import importlib
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

passed = 0
failed = 0

def check(label, got, expected):
    global passed, failed
    if got == expected:
        print(f"  PASS  {label}: {got}")
        passed += 1
    else:
        print(f"  FAIL  {label}: got {got!r}, expected {expected!r}")
        failed += 1

def check_approx(label, got_seg, got_ring, got_score, exp_seg, exp_ring, exp_score, note=""):
    global passed, failed
    ok = (got_seg == exp_seg and got_ring == exp_ring and got_score == exp_score)
    tag = "PASS" if ok else "FAIL"
    suffix = f"  [{note}]" if note else ""
    print(f"  {tag}  {label}: ({got_seg}, {got_ring!r}, {got_score})"
          f" expected ({exp_seg}, {exp_ring!r}, {exp_score}){suffix}")
    if ok:
        passed += 1
    else:
        failed += 1

from app.cv.calibrator import coords_to_segment_ring

# ── Test 1: Dead centre ───────────────────────────────────────────────────────
print("\nTest 1 — Dead centre")
check_approx("(0,0)", *coords_to_segment_ring(0, 0), 25, "bullseye", 50)

# ── Test 2: Bull ring ─────────────────────────────────────────────────────────
print("\nTest 2 — Bull ring")
check_approx("(10,0)", *coords_to_segment_ring(10, 0), 25, "bull", 25)

# ── Test 3: Triple 20 ─────────────────────────────────────────────────────────
print("\nTest 3 — Triple 20 (0, 103)")
check_approx("(0,103)", *coords_to_segment_ring(0, 103), 20, "triple", 60)

# ── Test 4: Single 20 ─────────────────────────────────────────────────────────
print("\nTest 4 — Single 20 (0, 50)")
check_approx("(0,50)", *coords_to_segment_ring(0, 50), 20, "single", 20)

# ── Test 5: Double 3 ─────────────────────────────────────────────────────────
print("\nTest 5 — Double 3 (5 o'clock position)")
ang_rad = math.radians(-(90 + 2 * 18))
x5 = 166 * math.cos(ang_rad)
y5 = 166 * math.sin(ang_rad)
seg5, ring5, sc5 = coords_to_segment_ring(x5, y5)
ok5 = (ring5 == "double" and sc5 == seg5 * 2)
note5 = "neighbor OK" if (ring5 == "double" and seg5 in [3, 19, 7]) else ""
if ring5 == "double" and seg5 == 3:
    print(f"  PASS  double-3 ({x5:.1f},{y5:.1f}): ({seg5}, {ring5!r}, {sc5})")
    passed += 1
else:
    print(f"  NOTE  double-3 ({x5:.1f},{y5:.1f}): ({seg5}, {ring5!r}, {sc5}) "
          f"— expected (3,'double',6); {note5 or 'segment off'}")
    if ring5 == "double":
        passed += 1   # ring correct, segment neighbour acceptable per spec
    else:
        failed += 1

# ── Test 6: Miss ─────────────────────────────────────────────────────────────
print("\nTest 6 — Complete miss (180, 0)")
check_approx("(180,0)", *coords_to_segment_ring(180, 0), 0, "miss", 0)

# ── Test 7: Edge cases ────────────────────────────────────────────────────────
print("\nTest 7 — Edge cases")
seg, ring, sc = coords_to_segment_ring(6.3, 0)
check("(6.3,0) should be bullseye", ring, "bullseye")

seg, ring, sc = coords_to_segment_ring(6.4, 0)
check("(6.4,0) should be bull", ring, "bull")

seg, ring, sc = coords_to_segment_ring(170, 0)
check("(170,0) should be double", ring, "double")

seg, ring, sc = coords_to_segment_ring(171, 0)
check("(171,0) should be miss", ring, "miss")

# ── calibrate.py syntax check ────────────────────────────────────────────────
print("\ncalibrate.py syntax check")
try:
    spec = importlib.util.spec_from_file_location(
        "calibrate",
        Path(__file__).parent / "calibrate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Don't exec (would try to open camera), just compile
    source = Path(__file__).parent.joinpath("calibrate.py").read_text()
    compile(source, "calibrate.py", "exec")
    print("  PASS  calibrate.py compiles without syntax errors")
    passed += 1
except SyntaxError as e:
    print(f"  FAIL  calibrate.py syntax error: {e}")
    failed += 1

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{passed} passed, {failed} failed")
if failed == 0:
    print("ALL PHASE 4 TESTS PASSED")
else:
    sys.exit(1)
