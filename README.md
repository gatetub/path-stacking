## About
Mapped Path In physics simulation

<img width="554" height="418" alt="plots1000_1 025_0 0564_0" src="https://github.com/user-attachments/assets/e15b1777-7e19-4e3c-94df-3ae249c45238" />

[Screencast from 04-11-26 06:38:44 PM IST.webm](https://github.com/user-attachments/assets/90778990-677b-4576-9754-0d5d8f5c2785)


## Setup
1. git clone this repo
2. run `docker build -t antfields:demo .` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run --rm -it -e DISPLAY=$DISPLAY --env="QT_X11_NO_MITSHM=1" -v /tmp/.X11-unix:/tmp/.X11-unix --volume="/home/exx/Documents/antfields-demo:/antfields" --gpus all antfields:demo /bin/bash` to start the docker container. Note that you need to change the path to your own path. BVH library is only needed during evaluation.
4. run `python main.py --enemy X Y Z` to start the training. `--enemy` is the
target location and is **required** — see the target section below. EXPLORATION
(default) trains with active learning in the gibson env; `--no_explore` selects
READ_FROM_COOKED_DATA, which trains from previously collected data.
```
# train model with default EXPLORATION mode
python main.py --enemy 0.25 -0.10 0.0

# train model with READ_FROM_COOKED_DATA mode (autodiscovers explored_data.npy:
# data/explored_data.npy if present, otherwise the newest Experiments/ run)
python main.py --no_explore --enemy 0.25 -0.10 0.0

# or name the cooked data explicitly
python main.py --no_explore --enemy 0.25 -0.10 0.0 \
    --cooked_data ./Experiments/08_21_02_00/explored_data.npy

# evaluate model (defaults to the newest checkpoint under ./Experiments,
# paired with the explored_data.npy from that same run)
python eval.py

# or name one explicitly
python eval.py --model ./Experiments/08_21_02_00/Model_Epoch_02050_ValLoss_2.987289e-02.pt --scale_factor 10
```

`eval.py` takes `--model`, `--experiments`, `--mesh`, `--data`, `--scale_factor`,
`--enemy`, `--n`, `--seed`, `--step_size` and `--tol`. Two things to keep in
step with the checkpoint being evaluated: `--scale_factor` (20 for new runs, 10
for the existing `Experiments/08_21_*` ones) and `--enemy`, which should be the
target that checkpoint was trained against. Neither changes the reported metrics
— the field is a function of planner coordinates — but a mismatch misreports the
geometry in any plot the model draws.

## Physics (TurtleBot execution)

By default the agent is a disembodied LiDAR origin: it teleports to each planned
viewpoint (`self.cur_view = nbv`), so every path executes perfectly and
collisions cost nothing. `--physics` replaces that with a differential-drive
TurtleBot simulated in pybullet, which must actually *drive* the planned path.
The pose it achieves — not the one that was commanded — becomes the next sensing
location, so tracking error, nonholonomic constraints and wall contacts feed
back into exploration.

```
# drive the planned paths with a simulated TurtleBot
python main.py --enemy 0.25 -0.10 0.0 --physics

# watch it in the pybullet GUI
python main.py --enemy 0.25 -0.10 0.0 --physics --physics_gui
```

Requires `pybullet` (already present in the iGibson base image) and a
`turtlebot.urdf`, autodetected from the iGibson assets or `pybullet_data`;
override with `--turtlebot_urdf`. If the simulator cannot start, the run prints
a warning and falls back to the original teleporting behaviour rather than
aborting.

pybullet loads `data/mesh.obj` at the same `scale_factor` as the renderer, so
physics and sensing agree about where the walls are. Sensing itself is
unchanged, so the learned speed field is unaffected by enabling physics.
`--sensor_height` is an **absolute z** in the unscaled frame (not an offset above
the floor, which sits at `z = -0.1196`). It defaults to `0.0`, mid-room, which
keeps the LiDAR exactly where the kinematic version had it. A real 0.5 m Kobuki
mast would be `z = -0.095` at `scale_factor=20`, but lowering it changes the
LiDAR view and therefore the training data.

### `scale_factor` is 20, because a TurtleBot does not fit at 10

`data/mesh.obj` is normalized to a 1.0 x 0.66 x 0.24 box, so `scale_factor` sets
the room's physical size. The mesh is cluttered, and at `scale_factor=10` the
free gaps are narrower than the robot:

| | `scale_factor=10` | `scale_factor=20` |
| --- | --- | --- |
| room | 10 x 6.6 m | 20 x 13.2 m |
| clearance at start `(-0.3,-0.2)` | 0.111 m | 0.222 m |
| clearance at Xe `(0.25,-0.10)` | 0.136 m | 0.272 m |
| TurtleBot footprint radius | 0.177 m | 0.177 m |
| floor admitting the chassis | ~48% | ~68% |
| verdict | **does not fit** | fits |

A zero-radius point agent threads gaps a real base cannot — the first thing
physics exposes. Startup measures clearance at both the start and the target and
warns; `--strict_clearance` makes it an error. If you need to go back to
`scale_factor=10` for an existing checkpoint, use `--robot_scale 0.6` to shrink
the robot instead; that keeps the sensing geometry untouched but the robot is no
longer TurtleBot-sized.

Note that the `0.543` coverage stop threshold was tuned for a zero-radius agent;
a footprint locked out of a third of the map may still not reach it.

### What `scale_factor` does to the speed data

**Speed values do not change.** Ground-truth speed is
`S* = clip(d, minimum, maximum) / maximum`, where `d` is distance-to-surface in
world units and `minimum`/`maximum` are multiplied by `scale_factor` at the point
of use. Numerator and denominator scale together, so `S*` at a given *planner*
coordinate is identical at 10 and at 20 — the function the network fits is the
same. The LiDAR range scales too (`max_depth` = 0.25 planner either way), so the
sensor sees the same fraction of the room.

**Two constants in the pair sampler did not scale**, and were fixed as part of
this change. `igib_runner.py` sampled the random start→goal separation as
`rand*1.5` and filtered pairs with `norm0 < 2.5`, both in *world* metres, while
everything around them scaled. Going 10 → 20 would therefore have silently
**halved** the planner-frame separation of the training pairs (0.15 → 0.075)
while `policy_occ` kept querying `T()` across the whole room — training on short
pairs and evaluating on long ones. They are now `PAIR_RADIUS = 0.15` and
`PAIR_MAX_RANGE = 0.25` in planner units, scaled at use. These values reproduce
the old numbers exactly at `scale_factor=10`, so the published baseline is
unchanged.

**One thing genuinely improves.** `minimum` is where speed saturates at the wall
floor — the field's notion of "too close". At `scale_factor=10` that is
`0.007 x 10 = 0.07 m`, well inside the 0.177 m robot radius, so the field happily
routed through space the base cannot occupy. At 20 it is `0.14 m`, within 21% of
the footprint, so the learned "too close" band now roughly coincides with the
robot's actual extent.

`eval.py` has its own `scale_factor` and must match whatever a checkpoint was
trained with.

### What physics reports

Per frame: commanded vs achieved pose, tracking error, distance driven, and
wall-contact onsets. At the end of the run: total distance, total contacts and
stuck events. Under teleporting these are all identically zero. The planned path
still accumulates into `traj_*.npy`; the path actually driven is saved separately
as `traj_executed_*.npy` so the two can be compared.

## Note

We use bvh-distance-queries library [repo](https://github.com/YuliangXiu/bvh-distance-queries?tab=readme-ov-file) to calculate the distance between two objects. If the bvh library fails to install in dockerfile, you can install it manually by "cd /antfields/bvh-distance-queries && pip install -e .".
Occupancy map is modified from [repo](https://github.com/richardos/occupancy-grid-a-star) to fit the igibson env.
Data sampling and preprocessing is modified from [repo](https://github.com/facebookresearch/iSDF)

## License

This project is licensed under the Purdue University Non-Commercial Open Source Software License.  
See the [LICENSE](./LICENSE.md) file for details.  
For commercial use rights, please contact the Purdue Research Foundation Office of Technology Commercialization (otcip@prf.org).
