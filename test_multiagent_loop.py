"""Run the REAL train() loop against a fake world.

Agent and train() are pulled out of models/model_igibson.py with ast, so this
exercises the shipped control flow: sensing per agent, the shared map update,
the termination rules and the agent bookkeeping. torch / igibson / the PINN are
stubbed; only numpy is required.
"""
import ast, io, sys, types, contextlib
import numpy as np

# ----------------------------- fake torch -----------------------------
class FT:                      # minimal tensor: a numpy array with .clone() etc
    def __init__(self, a):
        a = np.asarray(a)
        self.a = a if a.dtype.kind in 'bi' else a.astype(np.float32)
    def clone(self):   return FT(self.a.copy())
    def detach(self):  return self
    def cpu(self):     return self
    def to(self, *_, **__): return self
    def numpy(self):   return self.a
    def item(self):    return float(self.a.reshape(-1)[0])
    def __getitem__(self, i):
        if isinstance(i, FT):
            i = i.a
        if isinstance(i, tuple):
            i = tuple(x.a if isinstance(x, FT) else x for x in i)
        r = self.a[i]
        return FT(r) if isinstance(r, np.ndarray) else r
    def __setitem__(self, i, v): self.a[i] = v
    def __gt__(self, o):  return FT(self.a > (o.a if isinstance(o, FT) else o))
    def __lt__(self, o):  return FT(self.a < (o.a if isinstance(o, FT) else o))
    def __and__(self, o): return FT(self.a.astype(bool) & (o.a.astype(bool) if isinstance(o, FT) else o))
    def __len__(self):  return len(self.a)
    def unsqueeze(self, d): return FT(np.expand_dims(self.a, d))
    def view(self, *sh): return FT(self.a.reshape(*sh))
    @property
    def shape(self): return self.a.shape
    @property
    def device(self): return 'cpu'
    def __repr__(self): return "FT(%s)" % np.round(self.a, 3).tolist()

def _cat(xs, dim=0):
    return FT(np.concatenate([x.a for x in xs], axis=dim))

torch = types.SimpleNamespace(
    cat=_cat,
    is_tensor=lambda x: isinstance(x, FT),
    randperm=lambda n: np.random.permutation(n),
    no_grad=contextlib.contextmanager(lambda: iter([None])),
    randint=lambda lo, hi, sh, device=None: FT(np.random.randint(lo, hi, sh)),
)
Tensor = FT
EXPLORATION, READ_FROM_COOKED_DATA = 1, 2

# ------------------------ pull the real code out ------------------------
SRC = io.open("models/model_igibson.py", encoding="utf-8").read()
tree = ast.parse(SRC)
ns = {"np": np, "torch": torch, "Tensor": Tensor, "os": __import__("os"),
      "EXPLORATION": EXPLORATION, "READ_FROM_COOKED_DATA": READ_FROM_COOKED_DATA}
grabbed = set()
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "Agent":
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<x>", "exec"), ns)
        grabbed.add("Agent")
    if isinstance(node, ast.ClassDef) and node.name == "Model":
        for sub in node.body:
            if isinstance(sub, ast.FunctionDef) and sub.name in ("train", "active_agents"):
                sub.decorator_list = []
                exec(compile(ast.fix_missing_locations(
                    ast.Module(body=[sub], type_ignores=[])), "<x>", "exec"), ns)
                grabbed.add(sub.name)
assert grabbed == {"Agent", "train", "active_agents"}, grabbed
Agent = ns["Agent"]
ns["igib_runner"] = types.SimpleNamespace()

# ----------------------------- fake world -----------------------------
class FakeMap:
    def __init__(self, cov_per_frame, frontiers):
        self.cov = 0.0
        self.cov_per_frame = cov_per_frame
        self.frontiers = frontiers     # callable(frame) -> list of (x, y)
        self.updates = []              # (agent_pos, n_points) per update
        self.frame = 0
    def update(self, pos, pts, bounds):
        self.updates.append((tuple(np.round(pos, 3)), len(pts)))
    def get_coverage(self):
        return self.cov
    def get_block_centers(self):
        return np.asarray(self.frontiers(self.frame), dtype=np.float32)

class FakeModel:
    def __init__(self, n_agents=2, starts=None, frontiers=None,
                 cov_per_frame=0.05, coverage_limit=0.543, camera_steps=100,
                 reach_at_frame=None, route_at_frame=None):
        self.folder = "/tmp/__fake_never_written"
        self.mode = EXPLORATION
        self.n_agents = n_agents
        self.starts = (np.asarray(starts, np.float32) if starts is not None
                       else np.array([[-0.3, -0.2, 0.0], [0.3, 0.2, 0.0]][:n_agents],
                                     dtype=np.float32))
        self.agents = []
        self.occ_map = FakeMap(cov_per_frame,
                               frontiers or (lambda f: [(0.1, 0.1), (-0.1, -0.1),
                                                        (0.2, -0.2), (-0.2, 0.2)]))
        self.coverage_limit = coverage_limit
        self.camera_steps = camera_steps
        self.idle_limit = 3
        self.frame_points_per_agent = 10000
        self.frame_idx = 0
        self.epoch = 0
        self.alpha = 1.0
        self.baseline = False
        self.enemy_found = False
        self.arrival_radius = 0.05
        self.search_weight = 2.0
        self.speed_weight = 0.0
        self.sep_weight = 0.0
        self.minimum, self.maximum, self.scale_factor = 0.007, 0.07, 10
        self.dim = 3
        self.Params = {'Device': 'cpu'}
        self.all_framedata = None
        self.frame_buffer_size = 20
        self.trajectory = None
        self.cur_view = None
        self.initial_view = None
        self.Xe = np.array([0.25, -0.10, 0.0], dtype=np.float32)
        self.reach_at_frame = reach_at_frame
        self.route_at_frame = route_at_frame
        self.plots = 0
        self.train_calls = []
        self.sense_calls = []

    # --- stubs -----------------------------------------------------
    def load_rawdata(self):
        self.agents = [Agent(i, s) for i, s in enumerate(self.starts)]
        self.initial_view = self.agents[0].cur_view.clone()
        self.cur_view = self.agents[0].cur_view
    active_agents = ns["active_agents"]
    def apply_enemy_slowdown(self, p, s): return s
    def train_core(self, ep, fd, one):
        self.train_calls.append((fd.shape, one))
        self.epoch += ep
        if one:
            b = fd.unsqueeze(0)
            self.all_framedata = (b if self.all_framedata is None
                                  else _cat([self.all_framedata, b], 0))
        return FT([0.5]), FT(np.zeros((10, 8), np.float32))
    def has_reached_enemy(self, pos, radius=0.0):
        return (self.reach_at_frame is not None
                and self.frame_idx >= self.reach_at_frame)
    def is_enemy_reachable(self, pos):
        ok = (self.route_at_frame is not None
              and self.frame_idx >= self.route_at_frame)
        if ok:
            self.enemy_found = True
        return ok, None
    def path_to_enemy(self, pos, step_size=0.02):
        p = np.asarray(pos, np.float32)[:3]
        return np.linspace(p, self.Xe, 12).astype(np.float32)
    def assign_frontiers(self, agents, height):
        locs = self.occ_map.get_block_centers()
        if len(locs) == 0:
            return [None] * len(agents), None
        picks = [locs[i] if i < len(locs) else None for i in range(len(agents))]
        return picks, {'idx': list(range(len(agents)))}
    def plan_to_target(self, pos, target, height, step_size=0.05):
        p = np.asarray(pos, np.float32)[:3]
        t = np.concatenate([np.asarray(target, np.float32)[:2], [height]])
        traj = np.linspace(p, t, 9).astype(np.float32)
        return traj, 3, 0.05
    def plot(self, *a, **k): self.plots += 1
    def save(self, **k): pass
    train = ns["train"]

def fake_sense(model, pos, mn, mx, num=10000, scale_factor=1):
    model.sense_calls.append(tuple(np.round(np.asarray(pos)[:3], 3)))
    pts = np.zeros((num, 6), np.float32)
    pts[:, 3:5] = np.asarray(pos)[:2]
    spd = np.ones((num, 2), np.float32)
    bnd = np.full((num, 2), 0.05, np.float32)
    return FT(pts), FT(spd), FT(bnd)

ns["igib_runner"].sample_points_and_speeds_from_pos_new = fake_sense
ns["os"] = types.SimpleNamespace(path=types.SimpleNamespace(exists=lambda p: True),
                                 makedirs=lambda p: None)

fails = []
def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)

def run(**kw):
    m = FakeModel(**kw)
    # coverage grows each frame
    orig = m.occ_map.get_coverage
    def cov():
        return m.frame_idx * m.occ_map.cov_per_frame
    m.occ_map.get_coverage = cov
    m.train()
    return m

print("\n--- A. two agents, terminate on combined coverage ---")
m = run(n_agents=2, cov_per_frame=0.05)
check(len(m.agents) == 2, "two agents created")
check(m.frame_idx == 11, "stopped at frame %d (0.543/0.05 -> 11)" % m.frame_idx)
# per frame: 2 sense calls, 2 map updates, 1 train call
check(len(m.sense_calls) == 2 * (m.frame_idx + 1),
      "sensed once per agent per frame (%d calls)" % len(m.sense_calls))
check(len(m.occ_map.updates) == 2 * (m.frame_idx + 1),
      "map updated once per agent per frame (%d)" % len(m.occ_map.updates))
check(len(m.train_calls) == m.frame_idx + 2,
      "train_core: %d loop calls + 1 finetune" % (m.frame_idx + 1))
loop_calls = [c for c in m.train_calls if c[1]]
check(len(loop_calls) == m.frame_idx + 1,
      "ONE train_core call per frame (%d)" % len(loop_calls))
check(all(c[0] == (20000, 8) for c in loop_calls),
      "every frame is n_agents*10000 points, got %s"
      % sorted(set(c[0] for c in loop_calls)))
upd0 = m.occ_map.updates[0][0]
upd1 = m.occ_map.updates[1][0]
check(upd0 != upd1,
      "map updated with EACH agent's own position (%s vs %s)" % (upd0, upd1))
check(all(a.trajectory is not None and len(a.trajectory) > 0 for a in m.agents),
      "both agents accumulated a path")
p0 = m.agents[0].pos(); p1 = m.agents[1].pos()
check(not np.allclose(p0, p1), "agents ended at different places %s %s"
      % (p0.round(3).tolist(), p1.round(3).tolist()))

print("\n--- B. mission ends when ANY agent reaches Xe ---")
m = run(n_agents=2, cov_per_frame=0.001, reach_at_frame=4)
check(m.frame_idx == 4, "stopped at frame %d, not on coverage" % m.frame_idx)
check(any(a.reached for a in m.agents), "an agent is flagged reached")

print("\n--- C. no frontiers left -> all retire ---")
m = run(n_agents=2, cov_per_frame=0.001,
        frontiers=lambda f: [] if f >= 0 else [(0.1, 0.1)])
check(all(a.done for a in m.agents), "every agent retired")
check(m.frame_idx == 0, "stopped immediately (frame %d)" % m.frame_idx)

print("\n--- D. frame budget ---")
m = run(n_agents=2, cov_per_frame=0.0, camera_steps=7)
check(m.frame_idx == 7, "stopped at the budget (frame %d)" % m.frame_idx)

print("\n--- E. fewer frontiers than agents -> idle then retire ---")
m = run(n_agents=2, cov_per_frame=0.0, camera_steps=50,
        frontiers=lambda f: [(0.1, 0.1)])
check(m.agents[1].done, "starved agent retired")
check(not m.agents[0].done, "fed agent kept going")
check(m.agents[1].idle_count >= 3,
      "retired after %d idle frames (limit 3)" % m.agents[1].idle_count)
check(m.frame_idx == 50,
      "the surviving agent kept working to the budget (frame %d)" % m.frame_idx)
check(len([c for c in m.sense_calls]) < 2 * 51,
      "the retired agent stopped sensing (%d calls, 2 agents would be %d)"
      % (len(m.sense_calls), 2 * 51))
widths = sorted(set(c[0] for c in m.train_calls if c[1]))
check(widths == [(20000, 8)],
      "frame width constant at 20000 across the retirement, got %s" % widths)

print("\n--- F. route found -> both agents switch to APPROACH ---")
m = run(n_agents=2, cov_per_frame=0.0, camera_steps=30, route_at_frame=2,
        reach_at_frame=9)
check(m.enemy_found, "enemy_found armed")
check(m.frame_idx == 9, "ended on arrival at frame %d" % m.frame_idx)
d0 = np.linalg.norm(m.agents[0].pos()[:2] - m.Xe[:2])
d1 = np.linalg.norm(m.agents[1].pos()[:2] - m.Xe[:2])
s0 = np.linalg.norm(m.agents[0].start[:2] - m.Xe[:2])
s1 = np.linalg.norm(m.agents[1].start[:2] - m.Xe[:2])
check(d0 < s0 and d1 < s1,
      "both agents closed on Xe (%.3f<%.3f, %.3f<%.3f)" % (d0, s0, d1, s1))

print("\n--- G. n_agents=1 still works (back-compat) ---")
m = run(n_agents=1, starts=[[-0.3, -0.2, 0.0]], cov_per_frame=0.05)
check(len(m.agents) == 1, "one agent")
check(m.train_calls[0][0] == (10000, 8),
      "n=1 frame is 10000 points as before, got %s" % (m.train_calls[0][0],))
check(len(m.sense_calls) == m.frame_idx + 1, "one sense per frame")
check(m.cur_view is not None and m.trajectory is not None,
      "single-agent aliases cur_view / trajectory still populated")

print("\n%s" % ("ALL PASS" if not fails else "%d FAILURE(S):\n  %s"
                % (len(fails), "\n  ".join(fails))))
sys.exit(1 if fails else 0)
