## About
This is a minimal example of active learning ntfields with igibson setup. 

## Setup
1. git clone this repo
2. run `docker build -t antfields:demo .` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run --rm -it -e DISPLAY=$DISPLAY --env="QT_X11_NO_MITSHM=1" -v /tmp/.X11-unix:/tmp/.X11-unix --volume="/home/exx/Documents/antfields-demo:/antfields" --gpus all antfields:demo /bin/bash` to start the docker container. Note that you need to change the path to your own path. BVH library is only needed during evaluation.
4. run `python main.py` to start the training. In main.py, you can switch mode. READ_FROM_COOKED_DATA: train model with previous collected data, EXPLORATION: train model with active learning(exploration) in gibson env. 
```
# train model with default EXPLORATION mode
python main.py 

# train model with READ_FROM_COOKED_DATA mode
python main.py --no_explore

# evaluate model
python eval.py
```

## Multi-agent exploration

Two (or more) robots explore the **same** scene, sharing one PINN and one
occupancy map, and planning their own moves from it. Agent 0 keeps the
original start point, so `--n_agents 1` reproduces the single-agent behaviour
exactly.

```bash
# two agents from opposite corners (the default)
python main.py --enemy 0.25 -0.10 0.0 --n_agents 2

# explicit start positions: 3 coordinates per agent, unscaled frame
python main.py --enemy 0.25 -0.10 0.0 --n_agents 2 \
               --starts -0.3 -0.2 0.0   0.3 0.2 0.0

# single-agent A/B reference (original paper logic)
python main.py --enemy 0.25 -0.10 0.0 --n_agents 1 --baseline
```

### What is shared and what is not

| Shared (one copy) | Per agent |
| --- | --- |
| the PINN — it learns the *scene's* speed field | current position |
| the occupancy map — so coverage is already the union | walked trajectory |
| the frame buffer — observations are pooled | its frontier assignment |

This is **centralised information with per-agent planning**. One network is
not a simplification: it learns a property of the house, so training N
networks on 1/N of the frames each would give N worse models of the same
thing. Merging the agents' sensed points into one frame also produces
cross-agent point pairs, which is long-range supervision a single robot
cannot generate.

### Frontier assignment

Every agent scores frontiers with

```
score = T(x_i -> f_j) + search_weight * T(f_j -> Xe) + speed_weight * (1 - S(f_j))
```

The last two terms are identical for every agent, so independent `argmin`
makes both robots pick the *same* frontier. Instead the frontiers are
**assigned**: exact minimum-cost matching for 1 and 2 agents (no scipy
needed), Hungarian via scipy for N > 2 when available, otherwise sequential
greedy. Use `--sep_weight` if the agents still trail each other.

### Stopping

| Rule | Combines as |
| --- | --- |
| an agent reaches Xe | **any** — either robot arriving ends the run |
| combined coverage > `--coverage_limit` | nothing to combine: the shared map *is* the union |
| no frontiers left for an agent | that agent retires after 3 idle frames |
| every agent retired | **all** |
| frame budget | `--no_fair_budget` to disable |

By default the **total number of sensing actions** is held constant, so 2
agents get 50 frames where 1 agent gets 100. Without that, an N-agent run
simply receives N times the sensor budget and wins for the wrong reason.

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--n_agents` | 2 | number of robots |
| `--starts` | spread on a circle | `3 * n_agents` coordinates |
| `--sep_weight` | 0.0 | penalty for a frontier near another agent |
| `--sep_sigma` | 0.15 | length scale of that penalty |
| `--coverage_limit` | 0.543 | combined-coverage stop |
| `--no_fair_budget` | off | give each agent the full frame budget |
| `--baseline` | off | original paper logic, for A/B runs |

Start positions must be inside `-0.5..0.5` **and in free space** — that second
condition cannot be checked before the first render, so an agent started
inside geometry will produce garbage frames.

## Note

We use bvh-distance-queries library [repo](https://github.com/YuliangXiu/bvh-distance-queries?tab=readme-ov-file) to calculate the distance between two objects. If the bvh library fails to install in dockerfile, you can install it manually by "cd /antfields/bvh-distance-queries && pip install -e .".
Occupancy map is modified from [repo](https://github.com/richardos/occupancy-grid-a-star) to fit the igibson env.
Data sampling and preprocessing is modified from [repo](https://github.com/facebookresearch/iSDF)

## License

This project is licensed under the Purdue University Non-Commercial Open Source Software License.  
See the [LICENSE](./LICENSE.md) file for details.  
For commercial use rights, please contact the Purdue Research Foundation Office of Technology Commercialization (otcip@prf.org).
