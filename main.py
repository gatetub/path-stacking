# SYSTEMS AND METHODS FOR PHYSICS-INFORMED NEURAL NETWORKS FOR ROBOT MAPPING
# Copyright © 2025 Purdue University.
# Developed by Yuchen Liu, Ruiqi Ni and CORAL Lab.
# Purdue Research Foundation Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

import numpy as np
import argparse
from models import model_igibson as md

# Define modes
EXPLORATION = 1 
READ_FROM_COOKED_DATA = 2


# Define the model path
scale_factor = 10
modelPath = './Experiments'

def main():
    parser = argparse.ArgumentParser(description='Train the model with or without exploration.')
    parser.add_argument('--no_explore', action='store_true', help='Disable exploration mode.')
    #! Xe CHANGE: REQUIRED. Execution will not begin without a valid target
    #! coordinate inside the environment.
    parser.add_argument('--enemy', type=float, nargs=3, required=True,
                        metavar=('X', 'Y', 'Z'),
                        help='REQUIRED. Fixed target location inside the environment '
                             '(unscaled frame, within -0.5..0.5), e.g. --enemy 0.25 -0.10 0.0')
    parser.add_argument('--enemy_d_slow', type=float, default=0.30,
                        help='Radius of the slow-down zone around the target.')
    parser.add_argument('--enemy_s_min', type=float, default=0.15,
                        help='Speed multiplier at the target itself (0 < s_min <= 1).')
    #! Xe CHANGE: search priority + termination
    parser.add_argument('--search_weight', type=float, default=2.0,
                        help='Frontier score = T(robot->frontier) + w*T(frontier->Xe). '
                             'Higher = prioritise FINDING Xe over mapping. 0 = original sweep.')
    parser.add_argument('--arrival_radius', type=float, default=0.05,
                        help='Stop once the robot is within this distance of Xe.')
    parser.add_argument('--speed_weight', type=float, default=1.0,
                        help='Preference for high predicted speed at the chosen frontier '
                             '(open space). 0 = ignore speed.')
    #! MULTI-AGENT -------------------------------------------------------
    parser.add_argument('--n_agents', type=int, default=2,
                        help='Number of robots exploring the SAME map with the '
                             'SAME network. 1 reproduces the original '
                             'single-agent behaviour exactly.')
    parser.add_argument('--starts', type=float, nargs='*', default=None,
                        metavar='C',
                        help='Flat list of 3*n_agents start coordinates, e.g. '
                             '--starts -0.3 -0.2 0  0.3 0.2 0. Must be inside '
                             '-0.5..0.5 AND in free space. Omit to spread the '
                             'agents evenly round the circle through the '
                             "paper's start point.")
    parser.add_argument('--sep_weight', type=float, default=0.0,
                        help='Penalty for picking a frontier near ANOTHER '
                             'agent. 0 = off (the assignment already stops two '
                             'agents sharing a frontier). Try 0.1-0.5 if the '
                             'agents trail each other.')
    parser.add_argument('--sep_sigma', type=float, default=0.15,
                        help='Length scale of the separation penalty.')
    parser.add_argument('--coverage_limit', type=float, default=0.543,
                        help='Stop once combined coverage exceeds this. '
                             'Scene-specific safety net.')
    parser.add_argument('--no_fair_budget', action='store_true',
                        help='Give each agent the full frame budget instead of '
                             'holding TOTAL sensing actions constant. Off by '
                             'default so an N-agent run is comparable with '
                             '--baseline rather than just getting N times the '
                             'sensor budget.')
    #! A/B comparison ----------------------------------------------------
    parser.add_argument('--baseline', action='store_true',
                        help='Disable every Xe behaviour (frontier ranking, '
                             'approach phase, speed shaping) and terminate on '
                             'coverage only - the original paper logic.')
    parser.add_argument('--use_slowdown', action='store_true',
                        help='Enable the deceleration zone around Xe once a '
                             'route exists.')

    args = parser.parse_args()

    #! Validate the start positions before any expensive setup.
    starts = args.starts
    if starts is not None and len(starts) > 0:
        if len(starts) != 3 * args.n_agents:
            parser.error(
                '--starts needs exactly {} values for {} agents (3 per agent), '
                'got {}'.format(3 * args.n_agents, args.n_agents, len(starts)))
        starts = np.asarray(starts, dtype=np.float32).reshape(args.n_agents, 3)
    else:
        starts = None

    # Set the mode based on the --no_explore flag
    if args.no_explore:
        mode = READ_FROM_COOKED_DATA
    else:
        mode = EXPLORATION  # Default to exploration mode

    renderer = None
    if mode in [EXPLORATION]:
        from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
        meshpath = "data/mesh.obj"
        
        renderer = MeshRenderer(width=1200, height=680)
        renderer.load_object(meshpath, scale=np.array([1, 1, 1]) * scale_factor)
        renderer.add_instance_group([0])
        camera_pose = np.array([0, 0, 1])    
        view_direction = np.array([1, 0, 0])
        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 0])
        renderer.set_fov(90)

    # Initialize and train the model
    #! Xe CHANGE: the target is a REQUIRED constructor argument, validated
    #! there. An out-of-bounds or malformed Xe raises immediately, before any
    #! training work begins, so a [0,0,0] placeholder can never be used.
    model = md.Model(modelPath, 3, scale_factor, mode, args.enemy,
                     renderer, device='cuda:0',
                     enemy_d_slow=args.enemy_d_slow,
                     enemy_s_min=args.enemy_s_min,
                     search_weight=args.search_weight,
                     arrival_radius=args.arrival_radius,
                     speed_weight=args.speed_weight,
                     baseline=args.baseline,
                     use_slowdown=args.use_slowdown,
                     n_agents=args.n_agents,
                     starts=starts,
                     sep_weight=args.sep_weight,
                     sep_sigma=args.sep_sigma,
                     coverage_limit=args.coverage_limit,
                     fair_budget=not args.no_fair_budget)
    model.train()

if __name__ == '__main__':
    main()