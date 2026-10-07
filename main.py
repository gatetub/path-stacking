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
    
    args = parser.parse_args()

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
                     speed_weight=args.speed_weight)
    model.train()

if __name__ == '__main__':
    main()