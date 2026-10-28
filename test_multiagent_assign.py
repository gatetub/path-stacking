"""Exercise the ACTUAL shipped solve_assignment / default_starts.

Pulled out of models/model_igibson.py with ast so we test the real code,
not a copy. Only numpy is needed, so this runs outside the docker image.
"""
import ast, io, itertools, sys
import numpy as np

SRC = io.open("models/model_igibson.py", encoding="utf-8").read()
tree = ast.parse(SRC)

wanted = {"solve_assignment", "default_starts"}
ns = {"np": np}
found = set()
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        node.decorator_list = []           # drop @staticmethod
        mod = ast.Module(body=[node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod), "<x>", "exec"), ns)
        found.add(node.name)
assert found == wanted, "missing: %s" % (wanted - found)
solve = ns["solve_assignment"]
default_starts = ns["default_starts"]

fails = []
def check(cond, msg):
    if cond:
        print("  PASS  %s" % msg)
    else:
        print("  FAIL  %s" % msg)
        fails.append(msg)

def brute_pairs(score):
    """Independent optimal reference for 2 agents."""
    n, m = score.shape
    best, arg = np.inf, None
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            v = score[0, i] + score[1, j]
            if v < best:
                best, arg = v, (i, j)
    return best, arg

print("\n--- solve_assignment ---")

# 1. single agent
s = np.array([[5.0, 1.0, 3.0]])
check(solve(s) == [1], "n=1 picks the argmin")

# 2. two agents must get DISTINCT frontiers even when they rank identically
s = np.array([[1.0, 2.0, 9.0],
              [1.0, 2.0, 9.0]])
r = solve(s)
check(len(set(r)) == 2 and -1 not in r, "n=2 identical rows -> distinct picks %s" % r)
check(sorted(r) == [0, 1], "n=2 identical rows -> the two cheapest %s" % r)

# 3. two agents, optimality vs brute force over random matrices
rng = np.random.default_rng(0)
bad = 0
for t in range(300):
    m = int(rng.integers(2, 12))
    s = rng.random((2, m)) * 10
    r = solve(s)
    got = s[0, r[0]] + s[1, r[1]]
    best, _ = brute_pairs(s)
    if r[0] == r[1] or abs(got - best) > 1e-9:
        bad += 1
check(bad == 0, "n=2 optimal on 300 random matrices (%d mismatches)" % bad)

# 3b. greedy would be SUBOPTIMAL here; assignment must not be
#     agent0 slightly prefers f0; agent1 strongly needs f0.
s = np.array([[1.0, 1.1],
              [1.0, 50.0]])
r = solve(s)
check(r == [1, 0], "n=2 avoids the greedy trap, got %s (greedy would give [0,1])" % r)

# 4. fewer frontiers than agents
s = np.array([[1.0], [2.0]])
r = solve(s)
check(sorted(r) == [-1, 0], "n=2 m=1 -> one assigned, one unassigned %s" % r)

s = rng.random((3, 2))
r = solve(s)
check(r.count(-1) == 1, "n=3 m=2 -> exactly one unassigned %s" % r)
check(len(set(x for x in r if x >= 0)) == 2, "n=3 m=2 -> assigned are distinct %s" % r)

# 5. more agents than 2, all served
s = rng.random((4, 7))
r = solve(s)
check(-1 not in r and len(set(r)) == 4, "n=4 m=7 -> 4 distinct picks %s" % r)

# 6. no frontiers at all
check(solve(np.zeros((2, 0))) == [-1, -1], "m=0 -> all unassigned")

# 7. non-finite costs must not win (they are pre-mapped to 1e6 upstream,
#    but confirm a huge cost is never preferred)
s = np.array([[1e6, 2.0, 3.0],
              [1e6, 2.5, 1.0]])
r = solve(s)
check(0 not in r, "huge costs are not selected %s" % r)

print("\n--- default_starts ---")
d = default_starts(1, 3)
check(d.shape == (1, 3), "n=1 shape %s" % (d.shape,))
check(np.allclose(d[0], [-0.3, -0.2, 0.0], atol=1e-5),
      "n=1 reproduces the paper start %s" % d[0].round(4).tolist())

d = default_starts(2, 3)
check(d.shape == (2, 3), "n=2 shape %s" % (d.shape,))
check(np.allclose(d[0], [-0.3, -0.2, 0.0], atol=1e-5),
      "n=2 agent0 keeps the paper start %s" % d[0].round(4).tolist())
check(np.allclose(d[1], [0.3, 0.2, 0.0], atol=1e-5),
      "n=2 agent1 is diametrically opposite %s" % d[1].round(4).tolist())
check(float(np.linalg.norm(d[0] - d[1])) > 0.5,
      "n=2 starts are well separated (%.4f)" % np.linalg.norm(d[0] - d[1]))

for n in (1, 2, 3, 4, 6):
    d = default_starts(n, 3)
    inb = np.all(d >= -0.5) and np.all(d <= 0.5)
    dists = [np.linalg.norm(d[i] - d[j])
             for i, j in itertools.combinations(range(n), 2)]
    sep = (min(dists) > 0.05) if dists else True
    check(bool(inb and sep), "n=%d in bounds and separated" % n)

print("\n%s" % ("ALL PASS" if not fails else "%d FAILURE(S): %s" % (len(fails), fails)))
sys.exit(1 if fails else 0)
