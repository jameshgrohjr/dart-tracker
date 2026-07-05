import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import init_db, create_player, create_session
from app.game_engine.cricket import CricketGame
from app.game_engine.game_501 import Game501

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

init_db()

# ── Test 1: Cricket basic marks ──────────────────────────────────────────────
print("\nTest 1 — Cricket basic marks")
p1 = create_player("T1_Alice")
sid = create_session("cricket", [p1])
players = [{"id": p1, "name": "T1_Alice"}]
g = CricketGame(sid, players)
r = g.process_throw(segment=20, ring="triple", score=60)
check("marks[p1][20] after triple", g.marks[p1][20], 3)
check("scored_marks after triple", r["scored_marks"], 3)
r2 = g.process_throw(segment=20, ring="single", score=20)
check("marks[p1][20] stays 3 after overflow single", g.marks[p1][20], 3)
check("scored_marks is 0 on overflow single", r2["scored_marks"], 0)

# ── Test 2: Cricket scoring points ───────────────────────────────────────────
print("\nTest 2 — Cricket scoring points")
p2a = create_player("T2_P1")
p2b = create_player("T2_P2")
sid2 = create_session("cricket", [p2a, p2b])
players2 = [{"id": p2a, "name": "T2_P1"}, {"id": p2b, "name": "T2_P2"}]
g2 = CricketGame(sid2, players2)

# P1 closes 20 (triple = 3 marks)
g2.process_throw(segment=20, ring="triple", score=60)
# Advance 2 more throws so P1's turn comes around again
g2.process_throw(segment=1, ring="single", score=1)   # P1 throw 2
g2.process_throw(segment=1, ring="single", score=1)   # P1 throw 3
# Now P2's three throws
g2.process_throw(segment=1, ring="single", score=1)
g2.process_throw(segment=1, ring="single", score=1)
g2.process_throw(segment=1, ring="single", score=1)
# Back to P1 — triple 20 again (3 overflow → 60 pts)
r3 = g2.process_throw(segment=20, ring="triple", score=60)
check("P1 gets 60 pts from overflow triple-20", g2.points[p2a], 60)

# Finish P1's round
g2.process_throw(segment=1, ring="single", score=1)
g2.process_throw(segment=1, ring="single", score=1)
# P2 closes 20
g2.process_throw(segment=20, ring="triple", score=60)
g2.process_throw(segment=1, ring="single", score=1)
g2.process_throw(segment=1, ring="single", score=1)
# P1 throws at 20 again — now both closed, should get no extra points
pts_before = g2.points[p2a]
g2.process_throw(segment=20, ring="single", score=20)
check("P1 gets no pts when all players closed 20", g2.points[p2a], pts_before)

# ── Test 3: Cricket winner detection ─────────────────────────────────────────
print("\nTest 3 — Cricket winner detection")
p3 = create_player("T3_Solo")
sid3 = create_session("cricket", [p3])
g3 = CricketGame(sid3, [{"id": p3, "name": "T3_Solo"}])
for num in [15, 16, 17, 18, 19, 20]:
    g3.process_throw(segment=num, ring="triple", score=num * 3)
# Bull: bullseye=2 marks, then bull=1 mark → total 3
g3.process_throw(segment=25, ring="bullseye", score=50)
g3.process_throw(segment=25, ring="bull",     score=25)
check("check_winner returns p3 after all closed", g3.check_winner(), p3)

# ── Test 4: 501 basic ────────────────────────────────────────────────────────
print("\nTest 4 — 501 basic flow")
p4 = create_player("T4_Bob")
sid4 = create_session("501", [p4])
g4 = Game501(sid4, [{"id": p4, "name": "T4_Bob"}])
g4.process_throw(segment=20, ring="triple", score=60)
check("scores[p4] after triple-20", g4.scores[p4], 441)

# ── Test 5: 501 bust ─────────────────────────────────────────────────────────
print("\nTest 5 — 501 bust logic")
p5 = create_player("T5_Bust")
sid5 = create_session("501", [p5])
g5 = Game501(sid5, [{"id": p5, "name": "T5_Bust"}])
# Reduce to exactly 40 (501 - 461 = 40)  using 461 pts of throws
g5.scores[p5] = 40  # force remaining to 40 for test isolation
r5 = g5.process_throw(segment=20, ring="triple", score=60)
check("bust=True when score would go negative", r5["bust"], True)
check("score unchanged after bust", g5.scores[p5], 40)

# ── Test 6: 501 double-out ───────────────────────────────────────────────────
print("\nTest 6 — 501 double-out")
p6 = create_player("T6_Win")
sid6 = create_session("501", [p6])
g6 = Game501(sid6, [{"id": p6, "name": "T6_Win"}], start_score=40)
r6 = g6.process_throw(segment=20, ring="double", score=40)
check("winner=True on double-out", r6["winner"], True)
check("scores[p6] == 0", g6.scores[p6], 0)
check("check_winner returns p6", g6.check_winner(), p6)

# ── Test 7: 501 must finish on double ────────────────────────────────────────
print("\nTest 7 — 501 must finish on double")
p7 = create_player("T7_NoDbl")
sid7 = create_session("501", [p7])
g7 = Game501(sid7, [{"id": p7, "name": "T7_NoDbl"}], start_score=40)
g7.process_throw(segment=20, ring="single", score=20)  # reduces to 20 (not a bust)
check("score reduces to 20 normally", g7.scores[p7], 20)
r7 = g7.process_throw(segment=20, ring="single", score=20)  # would reach 0 but not double
check("bust=True when reaching 0 without double", r7["bust"], True)
check("score unchanged (still 20) after no-double bust", g7.scores[p7], 20)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{passed} passed, {failed} failed")
if failed == 0:
    print("ALL PHASE 3 TESTS PASSED")
else:
    sys.exit(1)
