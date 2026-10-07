# SYSTEMS AND METHODS FOR PHYSICS-INFORMED NEURAL NETWORKS FOR ROBOT MAPPING
# Copyright © 2025 Purdue University.
# Developed by Yuchen Liu, Ruiqi Ni and CORAL Lab.
# Purdue Research Foundation Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

import matplotlib
import numpy as np
import math
import random
import time
import os
from datetime import datetime, timedelta

import torch
from torch.nn import Linear
from torch import Tensor
from torch.autograd import Variable, grad
from torchvision import transforms


import matplotlib
import matplotlib.pylab as plt

import pickle

from timeit import default_timer as timer

import igib_runner
import gridmap
from dataprocessing import isdf_sample, data_pc_validate, transform 
import json 
import cv2 

torch.backends.cudnn.benchmark = True

EXPLORATION = 1 
READ_FROM_COOKED_DATA = 2

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def sigmoid(input):
 
    return torch.sigmoid(10*input)

class Sigmoid(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return sigmoid(input) 

class DSigmoid(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return 10*sigmoid(input)*(1-sigmoid(input)) 

def sigmoid_out(input):
 
    return torch.sigmoid(0.1*input)

class Sigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return sigmoid_out(input) 

class DSigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return 0.1*sigmoid_out(input)*(1-sigmoid_out(input)) 

class DDSigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return 0.01*sigmoid_out(input)*(1-sigmoid_out(input))*(1-2*sigmoid_out(input))


class NN(torch.nn.Module):
    
    def __init__(self, device, dim ,B):#10
        super(NN, self).__init__()
        self.dim = dim

        h_size = 128 #512,256
        #input_size = 128
        #self.T=2

        self.B = B.T.to(device)
        print(B.shape)
        input_size = B.shape[0]
        #decoder

        self.scale = 10

        self.act = torch.nn.Softplus(beta=self.scale)#ELU,CELU

        #self.env_act = torch.nn.Sigmoid()#ELU
        #self.ddact = torch.nn.Sigmoid()-torch.nn.Sigmoid()*torch.nn.Sigmoid()
        self.actout = Sigmoid_out()#ELU,CELU

        #self.env_act = torch.nn.Sigmoid()#ELU

        self.nl1=3
        self.nl2=2

        self.encoder = torch.nn.ModuleList()
        self.encoder1 = torch.nn.ModuleList()
        #self.encoder.append(Linear(self.dim,h_size))
        
        self.encoder.append(Linear(2*input_size,h_size))
        self.encoder1.append(Linear(2*input_size,h_size))
        
        for i in range(self.nl1-1):
            self.encoder.append(Linear(h_size, h_size)) 
            self.encoder1.append(Linear(h_size, h_size)) 
        
        self.encoder.append(Linear(h_size, h_size)) 

        self.generator = torch.nn.ModuleList()
        self.generator1 = torch.nn.ModuleList()
        for i in range(self.nl2):
            self.generator.append(Linear(h_size, h_size)) 
            self.generator1.append(Linear(2*h_size, 2*h_size)) 
        
        self.generator.append(Linear(h_size,h_size))
        self.generator.append(Linear(h_size,1))
    
    def init_weights(self, m):
        
        if type(m) == torch.nn.Linear:
            stdv = (1. / math.sqrt(m.weight.size(1))/1.)*2
            #stdv = np.sqrt(6 / 64.) / self.T
            m.weight.data.uniform_(-stdv, stdv)
            m.bias.data.uniform_(-stdv, stdv)
    
    def input_mapping(self, x):
        w = 2.*np.pi*self.B
        x_proj = x @ w
        #x_proj = (2.*np.pi*x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)    #  2*len(B)

    def out(self, coords):
        
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        size = coords.shape[0]
        #! Xe CHANGE: bounded slices. Xp is (qs | qg | qe) = 3*dim wide.
        #! The network itself still only consumes (qs, qg): the enemy influences
        #! the field through the SPEED it was trained on, not through the input.
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:2*self.dim]
        
        x = torch.vstack((x0,x1))

        # x = x.unsqueeze(1)
        
        x = self.input_mapping(x)
        x  = torch.sin(self.encoder[0](x))
        for ii in range(1,self.nl1):
            #i0 = x
            x_tmp = x
            x  = torch.sin(self.encoder[ii](x))
            # x  = self.act(self.encoder1[ii](x) + x_tmp) 

        
        x = self.encoder[-1](x)

        x0 = torch.sin(x[:size,...])
        x1 = torch.sin(x[size:,...])

        # xx = torch.cat((x0, x1), dim=1)
        
        # x_0 = torch.logsumexp(self.scale*xx, 1)/self.scale
        # x_1 = -torch.logsumexp(-self.scale*xx, 1)/self.scale

        # x = torch.cat((x_0, x_1),1)
        x = (x0-x1)**2#torch.sqrt((x0-x1)**2 + 0.01)
        # x = torch.abs(x0-x1)
        for ii in range(self.nl2):
            x_tmp = x
            x = self.act(self.generator[ii](x)) 
            # x = self.act(self.generator1[ii](x) + x_tmp) 
            # x = torch.sin(self.generator[ii](x))
            # x = torch.sin(self.generator1[ii](x) + x_tmp)
        
        y = self.generator[-2](x)
        x = self.act(y)
        # x = torch.sin(y)

        y = self.generator[-1](x)
        x = self.actout(y)/math.e
        #print(x.shape)
        #output = output.squeeze(2)
        
        return x, coords

    # def forward(self, coords):
    #     coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input

    #     output, coords = self.out(coords)
    #     return output, coords


class Model():
    def __init__(self, ModelPath, dim, scale_factor, mode, Xe, renderer=None,
                 device='cpu', enemy_d_slow=0.30, enemy_s_min=0.30,
                 speed_floor=0.09, allow_origin_Xe=False,
                 search_weight=2.0, arrival_radius=0.0005, speed_weight=0.0,
                 use_slowdown=False, baseline=False):
        #! Xe CHANGE: the fixed target location is a REQUIRED argument and is
        #! validated immediately. There is no setter and no default: execution
        #! cannot begin until a valid in-bounds coordinate is supplied, so a
        #! [0,0,0] placeholder can never reach the rest of the code.

        # ======================= JSON Template =======================
        self.Params = {}
        self.Params['ModelPath'] = ModelPath
        # self.Params['DataPath'] = DataPath
        self.renderer = renderer
        self.dim = dim
        self.scale_factor = scale_factor
        current_time = datetime.utcnow()-timedelta(hours=5)
        self.folder = self.Params['ModelPath']+"/"+current_time.strftime("%m_%d_%H_%M")

        # Pass the JSON information
        self.Params['Device'] = device
        self.Params['Pytorch Amp (bool)'] = False

        self.Params['Network'] = {}
        self.Params['Network']['Normlisation'] = 'OffsetMinMax'

        self.Params['Training'] = {}
        self.Params['Training']['Number of sample points'] = 2e5
        self.Params['Training']['Batch Size'] = 2000
        self.Params['Training']['Validation Percentage'] = 10
        self.Params['Training']['Number of Epochs'] = 20000
        self.Params['Training']['Resampling Bounds'] = [0.1, 0.9]
        self.Params['Training']['Print Every * Epoch'] = 1
        self.Params['Training']['Save Every * Epoch'] = 50
        self.Params['Training']['Learning Rate'] = 1e-3#5e-5
        self.Params['Training']['Random Distance Sampling'] = True
        self.Params['Training']['Use Scheduler (bool)'] = False

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss = []
        self.epoch = 0
        self.frame_idx = 0
        self.trajectory = None
        self.prev_state_queue = []
        self.prev_optimizer_queue = []
        self.timer = []

        # # Parameters for the occupancy grid
        dim_cells = 100
        limit = 1.05
        spacing = limit/dim_cells
        self.occ_map = gridmap.OccupancyGridMap(np.zeros((dim_cells,dim_cells)), cell_size=spacing, occupancy_threshold=0.8, offset=limit/2)

        #! Xe CHANGE ---------------------------------------------------------
        #! Fixed target location. KNOWN to the agent from the start; the
        #! ENVIRONMENT (obstacles / free space) is what remains unknown.
        #!
        #! Valid world extent, used to validate Xe. Matches the plotting window
        #! (xmin/xmax = -0.5/0.5) and the occupancy map (limit 1.05). Widen if
        #! your mesh is larger.
        self.env_min = np.array([-0.5, -0.5, -0.5], dtype=np.float32)[:dim]
        self.env_max = np.array([ 0.5,  0.5,  0.5], dtype=np.float32)[:dim]

        #! Slow-down zone around Xe, expressed in the SPEED field (Eq.4 style).
        #! Within d_slow of Xe the ground-truth speed is scaled toward s_min, so
        #! the learned time field makes the robot decelerate as it closes in.
        #! Because this lives in S*, the network learns it and T / Gradient /
        #! path-following all stay mutually consistent.
        self.enemy_d_slow = float(enemy_d_slow)
        self.enemy_s_min  = float(enemy_s_min)
        self.enemy_found  = False
        #! Xe CHANGE / NaN FIX -----------------------------------------------
        #! Hard floor on the POST-slow-down ground-truth speed.
        #! The trainer applies  alpha*s^2*(2-s)^2 + (1-alpha)  with alpha>1, so
        #! (1-alpha) is NEGATIVE (-0.025 at alpha=1.025). Any speed below
        #! ~0.0814 makes that expression negative, and Loss then computes
        #! torch.sqrt(Yobs*Ypred) -> sqrt(negative) -> NaN, which permanently
        #! poisons the weights (the RepeatEpoch spin).
        #! The slow-down multiplier COMPOUNDS with obstacle proximity: a point
        #! near both a wall and Xe gets scaled twice, so it can easily cross
        #! that cliff. This floor keeps the product safely above it.
        self.speed_floor = float(speed_floor)

        #! Xe CHANGE / SEARCH PRIORITY ----------------------------------------
        #! Frontier ranking weight: score = T(robot->frontier)
        #!                                  + search_weight * T(frontier->Xe)
        #! 0.0 reproduces the paper's pure coverage sweep; larger values push
        #! the agent to prioritise FINDING Xe over mapping the whole scene.
        self.search_weight = float(search_weight)
        #! Weight on the SPEED preference. DEFAULT 0.0 - i.e. OFF.
        #!
        #! Why off: the Eikonal equation is ||grad T|| = 1/S, so along the
        #! optimal path  T = integral of (1/S) ds. The arrival time T is
        #! ALREADY the accumulated inverse speed. A frontier reachable only
        #! through slow, cluttered space already has a large T - the PINN
        #! optimises exactly this in train_core. Adding (1 - S) on top double
        #! counts speed information that T carries.
        #!
        #! It is also dimensionally inconsistent: T is a time, (1 - S) is a
        #! dimensionless number in [0,1], so their relative influence depends
        #! on the scale of the environment rather than on anything physical.
        #!
        #! Kept as a tunable in case you specifically want an endpoint
        #! "openness" bias for sensing quality, which is a perception argument
        #! rather than a planning one. Leave at 0.0 for the principled cost.
        self.speed_weight = float(speed_weight)
        #! Stop once the robot is this close to Xe.
        self.arrival_radius = float(arrival_radius)
        #! The slow-down zone is a REPELLENT during search: low speed = high
        #! arrival time, so frontiers near Xe score worse under argmin. We keep
        #! it disabled until a route to Xe exists, then enable it so the robot
        #! still decelerates on final approach.
        self.slowdown_active = False
        #! Xe CHANGE / DISABLED BY DEFAULT ------------------------------------
        #! The slow-down zone is OFF unless explicitly requested.
        #! Rationale: in a minimum-time formulation a low-speed region around
        #! the target is self-defeating - low speed means high arrival-time
        #! cost, so the field actively discourages entering exactly the place
        #! we want to reach. It also perturbs the ground-truth speeds that the
        #! obstacle-avoidance behaviour depends on. Finding Xe is achieved by
        #! the frontier ranking, not by the slow-down, so this stays off unless
        #! you opt in with use_slowdown=True.
        self.use_slowdown = bool(use_slowdown)
        #! Xe CHANGE / BASELINE ----------------------------------------------
        #! baseline=True disables EVERY Xe behaviour: frontier ranking falls
        #! back to argmin(T(robot->frontier)), termination reverts to coverage
        #! only, there is no approach phase and no speed shaping. The run is
        #! then byte-for-byte the original paper behaviour, for A/B comparison.
        self.baseline = bool(baseline)
        #! A route counts as reachable only if predicted speed stays at least
        #! this multiple of the obstacle floor along its entire length.
        #! 1.0 = touching an obstacle; higher demands more clearance.
        self.reach_speed_margin = 1.5
        if self.baseline:
            self.search_weight = 0.0
            self.speed_weight = 0.0
            self.use_slowdown = False
            print("[Xe] BASELINE MODE - all Xe behaviour disabled, "
                  "running original paper logic")
        #! -------------------------------------------------------------------
        #! -------------------------------------------------------------------
        #! -------------------------------------------------------------------

        #! Validate and assign Xe. An out-of-bounds or malformed Xe raises
        #! immediately rather than silently degrading to a [0,0,0] placeholder.
        if Xe is None:
            raise ValueError(
                "Xe is required: pass a valid target coordinate inside the "
                "environment, e.g. Model(..., Xe=[0.25, -0.10, 0.0]). "
                "Execution will not start without one.")
        else:
            Xe = np.asarray(Xe, dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(Xe)):
                raise ValueError(
                    "Xe contains NaN/Inf: {}".format(Xe))
            if Xe.shape[0] != dim:
                raise ValueError(
                    "Xe must have {} coordinates, got {}: {}".format(
                        dim, Xe.shape[0], Xe))
            #! Reject the all-zeros origin: it is the classic "forgot to set
            #! it" placeholder. Pass allow_origin_Xe=True if [0,0,0] really is
            #! the target you want.
            if (not allow_origin_Xe) and np.allclose(Xe, 0.0):
                raise ValueError(
                    "Xe is [0, 0, 0], which is the uninitialised placeholder "
                    "value and almost certainly not a real target. Pass a real "
                    "coordinate inside the environment, or set "
                    "allow_origin_Xe=True if the origin is genuinely intended.")
            if not (np.all(Xe >= self.env_min) and np.all(Xe <= self.env_max)):
                raise ValueError(
                    "Xe {} is outside the environment bounds {} .. {}. "
                    "Pass coordinates in the UNSCALED frame (the mesh loads at "
                    "scale_factor={}, but points are divided back down by it).".format(
                        [round(float(v), 4) for v in Xe],
                        [round(float(v), 2) for v in self.env_min],
                        [round(float(v), 2) for v in self.env_max],
                        scale_factor))
            if not (0.0 < self.enemy_s_min <= 1.0):
                raise ValueError("enemy_s_min must be in (0, 1], got {}".format(
                    self.enemy_s_min))
            if self.enemy_d_slow <= 0.0:
                raise ValueError("enemy_d_slow must be > 0, got {}".format(
                    self.enemy_d_slow))
            self.Xe = Xe
            self.has_enemy = True
            print("=" * 62)
            print("[Xe] TARGET ASSIGNED")
            print("[Xe]   location (x, y, z) : [{:.4f}, {:.4f}, {:.4f}]".format(
                *[float(v) for v in self.Xe]))
            print("[Xe]   env bounds         : {} .. {}  (validated, in bounds)".format(
                [round(float(v), 2) for v in self.env_min],
                [round(float(v), 2) for v in self.env_max]))
            print("[Xe]   slow-down radius   : {:.4f}".format(self.enemy_d_slow))
            print("[Xe]   speed at target    : {:.4f}x  (1.0 = full speed)".format(
                self.enemy_s_min))
            print("=" * 62)
        #! -------------------------------------------------------------------

        self.mode = mode
        self.frame_buffer_size = 20
        self.camera_steps = 5000//50
        self.minimum = 0.007 #0.02
        self.maximum = 0.07  #0.1
        self.all_framedata = None
        self.all_surf_pc = []
        self.free_pc = []

        self.init_network()
        

    
    #! Xe CHANGE -------------------------------------------------------------
    def enemy_speed_factor(self, pts):
        """Multiplicative slow-down factor in [s_min, 1] for points `pts`.

        Same clipped form as Eq.4, but centred on Xe instead of the obstacles:
            factor = clip(||q - Xe|| / d_slow, s_min, 1)
        -> 1.0 far from the enemy (normal speed)
        -> s_min at the enemy    (crawl)
        `pts` is (N, dim); returns (N, 1).
        """
        qe = torch.as_tensor(self.Xe, dtype=pts.dtype, device=pts.device)
        d = torch.norm(pts - qe.unsqueeze(0), dim=1, keepdim=True)
        return torch.clip(d / self.enemy_d_slow, self.enemy_s_min, 1.0)

    def apply_enemy_slowdown(self, points, speeds):
        """Scale ground-truth speeds down near Xe.

        points : (N, 2*dim) = (qs, qg)
        speeds : (N, 2)     = (S*(qs), S*(qg))
        Applied to the training data, so the PDE solution itself encodes the
        No-op when the model was constructed without an Xe.
        """
        #! Xe CHANGE: during the SEARCH phase the slow-down zone repels the
        #! agent (slow => high arrival time => frontier ranked worse), which
        #! fights the goal of finding Xe. Stay at full speed until a route to
        #! Xe exists; enable deceleration only for the final approach.
        #! Xe CHANGE: disabled entirely unless opted in. Ground-truth speeds
        #! are then EXACTLY the paper's, so obstacle avoidance is unaffected.
        if not self.use_slowdown:
            return speeds
        if not self.slowdown_active:
            print("[Xe] slow-down INACTIVE (searching) - speeds unmodified")
            return speeds

        qs = points[:, :self.dim]
        qg = points[:, self.dim:2*self.dim]
        f = torch.cat((self.enemy_speed_factor(qs),
                       self.enemy_speed_factor(qg)), dim=1)
        #! Xe CHANGE / COLLISION FIX -----------------------------------------
        #! Raw speeds live in [wall_floor, 1.0] where
        #!     wall_floor = minimum/maximum  (0.007/0.07 = 0.1)
        #! and wall_floor IS the "this is an obstacle" value. Obstacle
        #! avoidance depends entirely on the CONTRAST between wall_floor and
        #! open space (10x by default).
        #!
        #! The previous version did  clamp(speeds * f, min=speed_floor).
        #! That was wrong: multiplying then clamping UPWARD raised wall speeds
        #! to the floor, so near Xe a wall (0.1) and open space (1.0) both
        #! became 0.15 - identical. The obstacle signal was erased inside the
        #! slow-down zone and planned paths drove through walls.
        #!
        #! Correct form: scale only the headroom ABOVE the wall value, so the
        #! wall value is never touched and contrast is preserved:
        #!     out = wall_floor + (speed - wall_floor) * f
        #! Walls stay exactly at wall_floor; open space slows toward it.
        #! The output can never drop below wall_floor (0.1), which is already
        #! above the NaN cliff (0.0814), so no upward clamp is needed.
        wall_floor = float(self.minimum) / float(self.maximum)
        out = wall_floor + (speeds - wall_floor) * f

        #! Defensive only: min() ensures this can never RAISE a wall speed,
        #! even if minimum/maximum are retuned.
        out = torch.clamp(out, min=min(self.speed_floor, wall_floor), max=1.0)

        #! Xe CHANGE: confirm Xe reached the training data. Printed once per
        #! frame; `n_slowed` counts sample points inside the slow-down zone.
        #! If that stays 0, Xe is nowhere near where the robot is sampling
        #! (most likely a scale_factor / coordinate-frame mismatch).
        n_slowed = int((f < 1.0).sum().item())
        #! contrast = ratio of fastest to slowest speed still visible to the
        #! network. If this collapses toward 1.0, obstacles become invisible
        #! and paths will cut through walls.
        contrast = float(out.max().item()) / max(float(out.min().item()), 1e-6)
        print("[Xe] slow-down @ [{:.3f}, {:.3f}, {:.3f}] | {}/{} pts in zone | "
              "factor x[{:.3f}, {:.3f}] | S* now [{:.4f}, {:.4f}] | "
              "wall floor {:.3f} preserved | contrast {:.1f}x".format(
                  float(self.Xe[0]), float(self.Xe[1]), float(self.Xe[2]),
                  n_slowed, f.numel(),
                  float(f.min().item()), float(f.max().item()),
                  float(out.min().item()), float(out.max().item()),
                  wall_floor, contrast))
        if contrast < 2.0:
            print("[Xe] WARNING: speed contrast {:.1f}x is very low - obstacles "
                  "may be invisible to the network. Raise enemy_s_min.".format(contrast))
        return out

    def append_enemy(self, XP):
        """Widen (N, 2*dim) -> (N, 3*dim) by appending qe to every row.

        Accepts torch or numpy; returns the same type. Already-3*dim input is
        passed through, so old 2*dim call sites keep working.
        """
        if XP.shape[1] == 3 * self.dim:
            return XP
        assert XP.shape[1] == 2 * self.dim, \
            "expected 2*dim or 3*dim columns, got {}".format(XP.shape[1])
        #! Xe CHANGE: without a target there is no qe block to append; pad with
        #! Xe is guaranteed valid by the constructor, so qe is always real.
        if torch.is_tensor(XP):
            qe = torch.as_tensor(self.Xe, dtype=XP.dtype, device=XP.device)
            return torch.cat((XP, qe.unsqueeze(0).expand(XP.shape[0], -1)), dim=1)
        qe = np.broadcast_to(self.Xe, (XP.shape[0], self.dim))
        return np.concatenate((XP, qe), axis=1)

    def enemy_arrival_time(self, q_from):
        """Arrival time from each row of `q_from` (N, dim) to the known Xe.

        This is the search primitive: the agent knows WHERE Xe is but not HOW
        to reach it through the partially-mapped environment. The PDE supplies
        the reachable, collision-free time-to-enemy.
        """
        q_from = torch.as_tensor(q_from, dtype=torch.float32)
        if q_from.dim() == 1:
            q_from = q_from.unsqueeze(0)
        qe = torch.as_tensor(self.Xe, dtype=torch.float32).unsqueeze(0)
        qe = qe.expand(q_from.shape[0], -1)
        XP = torch.cat((q_from, qe), dim=1).to(self.Params['Device'])
        return self.TravelTimes(XP)

    def has_reached_enemy(self, current_location, radius=0.0005):
        """ARRIVED check: is the robot physically at the target?

        Pure geometry on the known Xe coords - use this to end the episode.
        """
        q = np.asarray(current_location, dtype=np.float32).reshape(-1)[:self.dim]
        d = float(np.linalg.norm(q - self.Xe))
        reached = bool(d <= radius)
        #! Xe CHANGE: announce arrival
        if reached:
            print("=" * 62)
            print("[Xe] ARRIVED AT TARGET  (dist {:.4f} <= radius {:.4f})".format(
                d, radius))
            print("=" * 62)
        return reached

    def is_enemy_reachable(self, current_location, step_size=0.02, tol=0.03,
                           max_iter=200):
        """REACHABLE check: does a collision-free route to Xe exist yet?

        Runs the Eq.11 bidirectional descent and reports whether it converged.
        This is the meaningful "target found" trigger for this setup: the agent
        always knows WHERE Xe is, so what it discovers through exploration is
        whether it can GET there through the mapped free space.

        Returns (reachable: bool, traj: np.ndarray).
        Note this costs a full descent, so call it every N frames, not every
        gradient step.
        """
        src = torch.from_numpy(
            np.asarray(current_location, np.float32).reshape(-1)[:self.dim]).float()
        tar = torch.from_numpy(np.asarray(self.Xe, np.float32)).float()
        traj, converged = self.predict_trajectory2(
            src, tar, step_size=step_size, tol=tol,
            max_iter=max_iter, return_status=True)

        #! Xe CHANGE / REACHABILITY FIX --------------------------------------
        #! `converged` only means the two ends of the bidirectional descent
        #! MET. It says nothing about whether the path is collision-free, or
        #! whether it crosses territory the robot has actually sensed. The
        #! network is an MLP: it extrapolates a smooth field into completely
        #! unexplored space, so the descent converges from almost the first
        #! frame. That made enemy_found flip immediately, APPROACH take over,
        #! and policy_occ (with search_weight / speed_weight) never run again -
        #! which is exactly why tuning those weights changed nothing.
        #!
        #! A route is genuinely usable only if predicted speed stays above the
        #! obstacle floor along its whole length. Speed near wall_floor means
        #! the path is grazing or crossing an obstacle.
        path_ok, min_s = True, float('nan')
        if converged and len(traj) > 1:
            wall_floor = float(self.minimum) / float(self.maximum)
            thresh = self.reach_speed_margin * wall_floor
            wp = np.asarray(traj, dtype=np.float32)[:, :self.dim]
            XPw = np.zeros((len(wp), 3 * self.dim), dtype=np.float32)
            XPw[:, :self.dim] = np.asarray(
                current_location, np.float32).reshape(-1)[:self.dim]
            XPw[:, self.dim:2*self.dim] = wp
            XPw[:, 2*self.dim:3*self.dim] = self.Xe
            XPw = Variable(Tensor(XPw)).to(self.Params['Device'])
            s_path = self.Speed(XPw).to('cpu').data.numpy()
            s_path = np.nan_to_num(s_path, nan=0.0, posinf=1.0, neginf=0.0)
            min_s = float(s_path.min())
            path_ok = bool(min_s >= thresh)
            if not path_ok:
                print("[Xe] route rejected: min speed on path {:.4f} < {:.4f} "
                      "(grazes an obstacle) - continue exploring".format(
                          min_s, thresh))
        converged = bool(converged and path_ok)
        #! -------------------------------------------------------------------

        #! Xe CHANGE: announce the first time a route to Xe exists.
        if converged and not self.enemy_found:
            self.enemy_found = True
            #! Xe CHANGE: route exists -> arm the deceleration zone, but only
            #! if the feature was explicitly enabled.
            self.slowdown_active = bool(self.use_slowdown)
            print("=" * 62)
            print("[Xe] TARGET FOUND - route to Xe exists in the mapped space")
            print("[Xe]   Xe            : [{:.4f}, {:.4f}, {:.4f}]".format(
                *[float(v) for v in self.Xe]))
            print("[Xe]   from          : [{:.4f}, {:.4f}, {:.4f}]".format(
                *[float(v) for v in np.asarray(current_location).reshape(-1)[:self.dim]]))
            print("[Xe]   path waypoints: {}".format(len(traj)))
            print("[Xe]   min speed on path: {:.4f}".format(min_s))
            print("[Xe]   at epoch      : {}".format(self.epoch))
            print("=" * 62)
        elif not converged:
            print("[Xe] no route to target yet (descent hit {} iters) - "
                  "keep exploring".format(max_iter))
        return converged, traj

    #! Example wiring (left commented so your loop keeps its current control
    #! flow - drop this into the exploration loop when you want it):
    #!
    #!     if self.has_reached_enemy(self.cur_view.cpu().numpy()):
    #!         print("[Xe] arrived at target"); break
    #!     reachable, traj = self.is_enemy_reachable(self.cur_view.cpu().numpy())
    #!     if reachable:
    #!         print("[Xe] route found - switching to approach")
    #!         # follow `traj` instead of calling policy_occ this frame
    #!     else:
    #!         traj_list, idx = self.policy_occ(cur_loc, height)  # keep exploring

    def path_to_enemy(self, current_location, step_size=0.02):
        """Plan a path from current_location to the known Xe via Eq.11.

        Valid only once enough of the environment has been mapped for a
        collision-free route to exist; otherwise the gradient descent will
        stall and return a short/!incomplete trajectory.
        """
        src = torch.from_numpy(np.asarray(current_location, np.float32)).float()
        tar = torch.from_numpy(np.asarray(self.Xe, np.float32)).float()
        return self.predict_trajectory2(src, tar, step_size=step_size)
    #! ------------------------------------------------------------------------

    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y)                                                                 

        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x                                                                                                    
    
    def Loss(self, points, Yobs, beta, gamma):
        
      
        tau, Xp = self.network.out(points)
        dtau = self.gradient(tau, Xp)

        #! Xe CHANGE: bounded slices. The enemy is NOT injected into T here:
        #! it enters through the ground-truth speed Yobs (apply_enemy_slowdown),
        #! which keeps this hand-derived closed form of
        #! d/dq[ log(tau)^2 * ||qs-qg|| ] valid.
        D = Xp[:,self.dim:2*self.dim]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)#torch.norm(D, p=2, dim =1)**2
        
        
        DT0 = dtau[:,:self.dim]
        DT1 = dtau[:,self.dim:2*self.dim]

        T3    = tau[:,0]**2

        LogTau = torch.log(tau[:,0])
        LogTau2 = LogTau**2
        LogTau3 = LogTau**3
        LogTau4 = LogTau**4

        #print(tau.shape)

        T01    = 4*LogTau2*T0/T3*torch.einsum('ij,ij->i', DT0, DT0)
        T02    = -4*LogTau3/tau[:,0]*torch.einsum('ij,ij->i', DT0, D)
        #print(T02.shape)
        T11    = 4*LogTau2*T0/T3*torch.einsum('ij,ij->i', DT1, DT1)
        T12    = 4*LogTau3/tau[:,0]*torch.einsum('ij,ij->i', DT1, D)

        lap0 = 0.5*torch.einsum('ij,ij->i', DT0, DT0)#ltau[:,:self.dim].sum(-1) 
        lap1 = 0.5*torch.einsum('ij,ij->i', DT1, DT1)#ltau[:,self.dim:].sum(-1) 
        
        
        
        S0 = (T01+T02+LogTau4)
        S1 = (T11+T12+LogTau4)
       
        #0.001
        Ypred0 = torch.sqrt(S0)#torch.sqrt
        Ypred1 = torch.sqrt(S1)#torch.sqrt


        Ypred0_visco = Ypred0#
        Ypred1_visco = Ypred1#+gamma*lap1

        sq_Ypred0 = (Ypred0_visco)#+gamma*lap0
        sq_Ypred1 = (Ypred1_visco)#+gamma*lap1


        sq_Yobs0 = (Yobs[:,0])#**2
        sq_Yobs1 = (Yobs[:,1])#**2

        # loss0 = (sq_Yobs0/sq_Ypred0+sq_Ypred0/sq_Yobs0)#**2#+gamma*lap0
        # loss1 = (sq_Yobs1/sq_Ypred1+sq_Ypred1/sq_Yobs1)#**2#+gamma*lap1

        # diff = loss0 + loss1-4
        # loss_n = torch.sum((loss0 + loss1-4))/Yobs.shape[0]
        l0 = sq_Yobs0*(sq_Ypred0)
        l1 = sq_Yobs1*(sq_Ypred1)
        # l0 = torch.sqrt(torch.sqrt(l0))
        # l1 = torch.sqrt(torch.sqrt(l1))

        l0_2 = torch.sqrt(l0)
        l1_2 = torch.sqrt(l1)
        loss0 = (l0_2-1)**2#+gamma*lap0#**2
        loss1 = (l1_2-1)**2

        # loss0 = abs(l0-1)#+gamma*lap0#**2
        # loss1 = abs(l1-1)

        # loss0 = (l0+1/l0-2)#+gamma*lap0#**2
        # loss1 = (l1+1/l1-2)
        diff = loss0 + loss1
        loss_n = torch.sum((loss0 + loss1))/Yobs.shape[0]

        
        loss = beta*loss_n #+ 1e-4*(reg_tau)

        return loss, loss_n, diff

    def init_network(self):
        #! Initialising the network
        #self._init_network()
        self.B = 0.5 * torch.normal(0,1,size=(128,self.dim)) #0.5
        #torch.save(B, self.Params['ModelPath']+'/B.pt')

        self.network = NN(self.Params['Device'],self.dim, self.B)
        self.network.apply(self.network.init_weights)
        #self.network.float()
        self.network.to(self.Params['Device'])

        #! Defining the optimization scheme
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.Params['Training']['Learning Rate']
            ,weight_decay=0.1)
        if self.Params['Training']['Use Scheduler (bool)'] == True:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
            #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000,2000], gamma=0.5)

    def load_rawdata(self):
        #! load data
        initial_view = Tensor([-0.3, -0.2, 0])
        self.initial_view = initial_view
        
        if self.mode == READ_FROM_COOKED_DATA: # read from file
            # all_points = np.load("sampled_points.npy")
            # all_speeds = np.load("speed.npy")
            explored_data = np.load("data/explored_data.npy")
            self.explored_data = torch.tensor(explored_data).to(self.Params['Device'])

        self.cur_view = initial_view

    def randomize_data(self, frame_data):
        prev_frame_idx = torch.randperm(self.frame_idx).tolist()[:self.frame_buffer_size-1]
        prev_framedata = self.all_framedata[prev_frame_idx]
        chosen_framedata = torch.cat((prev_framedata, frame_data.unsqueeze(0)), dim=0).view(-1, 8)

        chosen_points = chosen_framedata[:,:6]
        chosen_speeds = chosen_framedata[:,6:]
        
        points = frame_data[:,:6]
        speeds = frame_data[:,6:]
        if False: # ray sampling method
            local_start_index = torch.randperm(points.shape[0])[:int(points.shape[0]*1.0)]
            local_end_index = torch.randperm(points.shape[0])[:int(points.shape[0]*1.0)]
            local_points_combination = torch.cat((points[local_start_index,:3], points[local_end_index,3:]), dim=1)
            local_speeds_combination = torch.cat((speeds[local_start_index,:1], speeds[local_end_index,1:]), dim=1)
        else: # random sampling method
            local_points_combination = torch.cat((points[:,:3], points[:,3:]), dim=1)
            local_speeds_combination = torch.cat((speeds[:,:1], speeds[:,1:]), dim=1)

        chosen_percent = 1.0
        # print("chosen points shape:", chosen_points.shape)
        chosen_start_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
        chosen_end_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
        # print("chosen points indice shape:", chosen_start_index.shape)
        chosen_start_points = chosen_points[chosen_start_index][:, :3]
        chosen_start_speeds = chosen_speeds[chosen_start_index][:, 0]
        chosen_end_points = chosen_points[chosen_end_index][:, 3:6]
        chosen_end_speeds = chosen_speeds[chosen_end_index][:, 1]
        # print("chosen start points shape:", chosen_start_points.shape)
        global_points_combination = torch.cat((chosen_start_points, chosen_end_points), dim=1)
        global_speeds_combination = torch.cat((chosen_start_speeds[:, None], chosen_end_speeds[:, None]), dim=1)

        all_points = torch.cat((local_points_combination, global_points_combination), dim=0)
        all_speeds = torch.cat((local_speeds_combination, global_speeds_combination), dim=0)

        cur_data = torch.cat((all_points, all_speeds), dim=1)
        return cur_data

    def train(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.load_rawdata()
        is_one_frame = True
        while True:
            if True:
                print("Current Viewpoint:", self.cur_view)

                #! generate data start
                if self.mode == EXPLORATION:
                    heights = [0]
                    curpoints = [] 
                    curspeeds = []
                    curbounds = []
                    for height in heights:
                        curviewpoint = self.cur_view.clone()
                        curviewpoint[2] = height
                        points, speeds, bounds = igib_runner.sample_points_and_speeds_from_pos_new(self, curviewpoint.cpu().numpy(), self.minimum, self.maximum, num=10000, scale_factor=self.scale_factor)
                        curpoints.append(points)
                        curspeeds.append(speeds)
                        curbounds.append(bounds)          
                    frame_points = torch.cat(curpoints, dim=0)
                    frame_speeds = torch.cat(curspeeds, dim=0)
                    frame_bounds = torch.cat(curbounds, dim=0)
                    #! Xe CHANGE: bake the slow-down zone around Xe into the
                    #! ground-truth speed, so the learned PDE encodes it.
                    frame_speeds = self.apply_enemy_slowdown(frame_points, frame_speeds)
                    frame_epoch = 50

                    #? save occupancy map
                    # self.occ_map.save(self.folder+"/occ_map.npz")
                elif self.mode == READ_FROM_COOKED_DATA:
                    if self.frame_idx >= self.explored_data.shape[0]:
                        break
                    heights = [0]
                    curpoints = [] 
                    curspeeds = []
                    curbounds = []         
                    frame_points = self.explored_data[self.frame_idx, :, :6]
                    frame_speeds = self.explored_data[self.frame_idx, :, 6:]
                    #! Xe CHANGE: same slow-down for pre-cooked data
                    frame_speeds = self.apply_enemy_slowdown(frame_points, frame_speeds)
                    frame_epoch = 50



                #! generate data end
                            
                frame_data = torch.cat((frame_points.to(self.Params['Device']), frame_speeds.to(self.Params['Device'])), dim=1)

            ####################! Core Training #############################
            self.alpha = 1.025
            total_diff, cur_data = self.train_core(frame_epoch, frame_data, is_one_frame)



            #! Policy
            if self.mode != READ_FROM_COOKED_DATA:
                # nbv = self.policy_greedy(pre_view, cur_data.detach(), gamma)
                # nbv = self.policy_handcrafted(frame_idx)
                traj_list = None
                if self.mode == EXPLORATION:
                    #! Update the occupancy grid
                    #? filter frame points with some z range
                    valid = (frame_points[:,-1] > -0.1) & (frame_points[:,-1] < 0.1)
                    self.occ_map.update(self.cur_view.clone().cpu().numpy(), frame_points[valid].cpu().numpy(), frame_bounds[valid].cpu().numpy())
                    coverage = self.occ_map.get_coverage()
                    print("Occupancy Grid Coverage:", coverage)

                    #! Xe CHANGE / TERMINATION: the mission is to reach Xe, not
                    #! to map the scene. Primary stop condition is arrival at
                    #! the target; coverage is kept only as a safety net so an
                    #! unreachable Xe cannot loop forever.
                    cur_np = self.cur_view.detach().clone().cpu().numpy()
                    #! Xe CHANGE: in baseline mode skip arrival + approach
                    #! entirely, so termination is coverage-only as in the paper.
                    if (not self.baseline) and self.has_reached_enemy(
                            cur_np, radius=self.arrival_radius):
                        print("[Xe] MISSION COMPLETE - target reached at epoch "
                              "{} (coverage {:.4f})".format(self.epoch, coverage))
                        break

                    #! Once a route to Xe exists, enable the deceleration zone
                    #! and head straight for the target instead of exploring.
                    if (not self.baseline) and (not self.enemy_found):
                        reachable, _ = self.is_enemy_reachable(cur_np)
                        if reachable:
                            print("[Xe] switching from SEARCH to APPROACH")

                    if coverage > 0.543:
                        if self.baseline:
                            print("Coverage limit reached - stopping (baseline).")
                        else:
                            print("[Xe] WARNING: coverage limit {:.3f} reached before "
                                  "arriving at Xe - stopping. The target may be "
                                  "unreachable or search_weight may be too low "
                                  "(currently {:.2f}).".format(coverage, self.search_weight))
                        break
                    #! Xe CHANGE: once a route to Xe exists, drive straight at
                    #! the target instead of picking another frontier.
                    if self.enemy_found:
                        traj_list = self.path_to_enemy(cur_np, step_size=0.02)
                        #! Xe CHANGE / STEP FIX: advance by accumulated PATH
                        #! DISTANCE, mirroring policy_occ, instead of a fixed
                        #! waypoint count. Waypoint spacing is proportional to
                        #! the local speed (Eq.11 step is ~ alpha*S), so a fixed
                        #! count of 4 moves a wildly variable - and near an
                        #! obstacle or inside a slow-down zone, tiny - distance.
                        #! A distance budget makes each frame advance the same
                        #! amount regardless of the local speed.
                        step_budget = 0.05
                        traj_ind = 0
                        accum_dis = 0.0
                        while accum_dis < step_budget and traj_ind < len(traj_list) - 1:
                            accum_dis += np.linalg.norm(
                                traj_list[traj_ind + 1][:2] - traj_list[traj_ind][:2])
                            traj_ind += 1
                        print("[Xe] APPROACH: {} waypoints | advanced {:.4f} "
                              "(budget {:.2f}) -> index {}".format(
                                  len(traj_list), accum_dis, step_budget, traj_ind))
                    else:
                        traj_list, traj_ind = self.policy_occ(
                            self.cur_view.detach().clone().cpu().numpy(), height=0)
                    nbv = Tensor(traj_list[traj_ind])


                # print("*"*10)
                # print("self.dataset.Ts[0][:3, 3]/self.scale_factor", self.dataset.Ts[0][:3, 3]/self.scale_factor)
                print("nbv", nbv)
                print("curview", self.cur_view)
                traj = self.predict_trajectory2(self.cur_view.detach().clone().cpu().numpy(), nbv.detach().clone().cpu().numpy())
                if self.mode == EXPLORATION:
                    traj = traj_list[:traj_ind+1]
                if self.trajectory is None:
                    self.trajectory = traj
                else:
                    self.trajectory = np.concatenate([self.trajectory, traj], axis=0) 
                print("traj is:", traj)
                
                
                #? ******************SAVINGS start*******************
                save_traj = False 
                if save_traj:
                    np.save(self.folder+"/traj"+"_"+str(self.epoch)+".npy", self.trajectory)


                
                camera_matrix = None
                self.plot(self.initial_view, nbv, self.epoch, total_diff.item(),self.alpha, cur_data[:,:6].clone().cpu().numpy(), camera_matrix, traj_list)

            elif self.mode == READ_FROM_COOKED_DATA:
                self.plot(self.initial_view, np.array([0.3, 0.2, 0]), self.epoch, total_diff.item(),self.alpha, cur_data[:,:6].clone().cpu().numpy(), None)

            
            # self.fourplot(epoch, FRAMES[:frame_idx+1], total_diff.item(), alpha)
            with torch.no_grad():
                self.save(epoch=self.epoch, val_loss=total_diff)

            #? ******************SAVINGS end*******************
            
            self.frame_idx += 1
            if self.frame_idx >= self.camera_steps:
                break
            
            if self.mode != READ_FROM_COOKED_DATA:
                self.cur_view = nbv

        if True:
            print("Exploration is done. Finetuning...")
            self.plot(self.cur_view, self.cur_view, self.epoch, total_diff.item(),self.alpha)
            #?: train for another 1000 epochs for finetuning
            # 1. data is all framedata
            prev_frame_idx = torch.randperm(self.frame_idx).tolist()[:self.frame_buffer_size-1]
            #! mix data so that the start and end points are from different frames
            prev_framedata = self.all_framedata[prev_frame_idx]
            chosen_percent = 1.0
            # print("chosen points shape:", chosen_points.shape)
            chosen_framedata = torch.cat((prev_framedata, frame_data.unsqueeze(0)), dim=0).view(-1, 8)

            chosen_points = chosen_framedata[:,:6]
            chosen_speeds = chosen_framedata[:,6:]
            chosen_start_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
            chosen_end_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
            # print("chosen points indice shape:", chosen_start_index.shape)
            chosen_start_points = chosen_points[chosen_start_index][:, :3]
            chosen_start_speeds = chosen_speeds[chosen_start_index][:, 0]
            chosen_end_points = chosen_points[chosen_end_index][:, 3:6]
            chosen_end_speeds = chosen_speeds[chosen_end_index][:, 1]
            # print("chosen start points shape:", chosen_start_points.shape)
            global_points_combination = torch.cat((chosen_start_points, chosen_end_points), dim=1)
            global_speeds_combination = torch.cat((chosen_start_speeds[:, None], chosen_end_speeds[:, None]), dim=1)
            fine_data = torch.cat((global_points_combination, global_speeds_combination), dim=1)

            # 2. core training
            fine_epoch = 1000
            total_diff, cur_data = self.train_core(fine_epoch, fine_data, False)

            # 3. save a final plot #
            self.plot(self.cur_view, self.cur_view, self.epoch, total_diff.item(),self.alpha)
            with torch.no_grad():
                self.save(epoch=self.epoch, val_loss=total_diff)

    def train_core(self, epoch, frame_data, is_one_frame=True):
        beta = 1.0
        prev_diff = 1.0
        current_diff = 1.0
        step = -2000.0/4000.0
        step = 0
        tt =time.time()

        if is_one_frame:
            if self.all_framedata is None:
                self.all_framedata = frame_data.unsqueeze(0)
            else:
                self.all_framedata = torch.cat((self.all_framedata, frame_data.unsqueeze(0)), dim=0)
            print(self.all_framedata.shape)
            np.save(f"{self.folder}/explored_data.npy", self.all_framedata.clone().cpu().numpy())


        #! mix data so that the start and end points are from different frames
        if is_one_frame:
            cur_data = self.randomize_data(frame_data)
            # print("cur_data:", cur_data.shape)
        else:
            rand_idx = torch.randperm(frame_data.shape[0])
            chosen_data = frame_data[rand_idx][:100000]
            chosen_points = chosen_data[:,:6]
            chosen_speeds = chosen_data[:,6:]
            
            chosen_percent = 1.0
            # print("chosen points shape:", chosen_points.shape)
            chosen_start_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
            chosen_end_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
            # print("chosen points indice shape:", chosen_start_index.shape)
            chosen_start_points = chosen_points[chosen_start_index][:, :3]
            chosen_start_speeds = chosen_speeds[chosen_start_index][:, 0]
            chosen_end_points = chosen_points[chosen_end_index][:, 3:6]
            chosen_end_speeds = chosen_speeds[chosen_end_index][:, 1]

            cur_data = torch.cat((chosen_start_points, chosen_end_points, chosen_start_speeds[:, None], chosen_end_speeds[:, None]), dim=1)
                
        # if self.mode == READ_FROM_COOKED_DATA: #? use all framedata
        dataloader = FastTensorDataLoader(cur_data, batch_size=int(self.Params['Training']['Batch Size']), shuffle=True)


        # frame_epoch = 10000 if isOffline else 50
        frame_epoch = epoch
        start_time = time.time()
        for e in range(frame_epoch):
            self.epoch += 1
            total_train_loss = 0
            total_diff=0

            # alpha = min(max(0.85,0.85+0.5*step),0.9)
            # alpha = 1.10#1.05
            

            step+=1.0/4000/((int)(self.epoch/4000)+1.) * 2.0
            gamma=0.001#max((4000.0-epoch)/4000.0/20,0.001)

            current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
            current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
            self.prev_state_queue.append(current_state)
            self.prev_optimizer_queue.append(current_optimizer)
            if(len(self.prev_state_queue)>5):
                self.prev_state_queue.pop(0)
                self.prev_optimizer_queue.pop(0)
            
            #self.optimizer.param_groups[0]['lr']  = np.clip(1e-3*(1-(epoch-8000)/1000.), a_min=5e-4, a_max=1e-3) 
            self.optimizer.param_groups[0]['lr']  = 5e-4


            prev_diff = current_diff
            iter=0
            while True:
                total_train_loss = 0
                total_diff = 0
                #for i in range(10):
                for i, data in enumerate(dataloader, 0):#train_loader_wei,dataloader
                    #print('----------------- Epoch {} - Batch {} --------------------'.format(epoch,i))
                    if i>5:
                        break
                    t0 = time.time()
    
                    data=data[0].to(self.Params['Device'])
                    #data, indexbatch = data
                    points = data[:,:2*self.dim]#.float()#.cuda()
                    speed = data[:,2*self.dim:]#.float()#.cuda()
                    
                    speed = speed*speed*(2-speed)*(2-speed)
                    speed = self.alpha*speed+1-self.alpha

                    #! Xe CHANGE / NaN FIX: alpha>1 makes (1-alpha) negative, so
                    #! this transform can output negatives for small speeds and
                    #! Loss's sqrt() then yields NaN. Clamp to a small positive
                    #! value and report, instead of failing silently.
                    n_neg = int((speed <= 0).sum().item())
                    if n_neg > 0:
                        print("[NaN guard] {} speed values <= 0 after the alpha "
                              "transform (min {:.6f}) - clamping to 1e-4. "
                              "Raise speed_floor if this recurs.".format(
                                  n_neg, float(speed.min().item())))
                        speed = torch.clamp(speed, min=1e-4)


                    loss_value, loss_n, wv = self.Loss(
                    points, speed, beta, gamma)
                    loss_value.backward()

                    # Update parameters
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    #print('')
                    #print(loss_value.shape)
                    total_train_loss += loss_value.clone().detach()
                    total_diff += loss_n.clone().detach()
                    t1 = time.time()
                    #print(t1-t0)
                    #print('')
                    #weights[indexbatch] = wv
                    
                    del points, speed, loss_value, loss_n, wv#,indexbatch
                

                total_train_loss /= 5 #dataloader train_loader
                total_diff /= 5 #dataloader train_loader

                current_diff = total_diff
                diff_ratio = current_diff/prev_diff
        
                if True and (diff_ratio < 1.2 and diff_ratio > 0 or e < 10):#1.5
                    #self.optimizer.param_groups[0]['lr'] = prev_lr 
                    break
                else:
                    
                    iter+=1
                    with torch.no_grad():
                        random_number = random.randint(0, 4)
                        self.network.load_state_dict(self.prev_state_queue[random_number], strict=True)
                        self.optimizer.load_state_dict(self.prev_optimizer_queue[random_number])

                    print("RepeatEpoch = {} -- Loss = {:.4e} -- Alpha = {:.4e}".format(
                        self.epoch, total_diff, self.alpha))
                
                
            #'''
            self.total_train_loss.append(total_train_loss)
            
            beta = 1.0/total_diff
            
            t_1=time.time()
            #print(t_1-t_0)

            #del train_loader_wei, train_sampler_wei

            if self.Params['Training']['Use Scheduler (bool)'] == True:
                self.scheduler.step(total_train_loss)

            t_tmp = tt
            tt=time.time()
            #print(tt-t_tmp)
            #print('')
            if self.epoch % self.Params['Training']['Print Every * Epoch'] == 0:
                with torch.no_grad():
                    #print("Epoch = {} -- Training loss = {:.4e} -- Validation loss = {:.4e}".format(
                    #    epoch, total_train_loss, total_val_loss))
                    print("Epoch = {} -- Loss = {:.4e} -- Alpha = {:.4e}".format(
                        self.epoch, total_diff.item(), self.alpha))
        end_time = time.time()
        # if is_one_frame:
        #     duration = end_time-start_time
        #     print(f"running in {duration} secs")
        #     self.timer.append(duration)
        #     np.save("gib6_time.npy",np.array(self.timer))
        return total_diff, cur_data

    def save(self, epoch='', val_loss=''):
        '''
            Saving a instance of the model
        '''
        torch.save({'epoch': epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'B_state_dict':self.B,
                    'train_loss': self.total_train_loss,
                    'val_loss': self.total_val_loss}, '{}/Model_Epoch_{}_ValLoss_{:.6e}.pt'.format(self.folder, str(epoch).zfill(5), val_loss))

    def load(self, filepath):
        #B = torch.load(self.Params['ModelPath']+'/B.pt')
        
        checkpoint = torch.load(
            filepath, map_location=torch.device(self.Params['Device']))
        self.B = checkpoint['B_state_dict']

        self.network = NN(self.Params['Device'],self.dim,self.B)

        self.network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.network.to(torch.device(self.Params['Device']))
        self.network.float()
        self.network.eval()

    def load_traj(self, filepath):
        self.trajectory = np.load(filepath)
     
    def Gradient(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
        #! Xe CHANGE: accept 2*dim or 3*dim input
        Xp = self.append_enemy(Xp)
       
        #Xp.requires_grad_()
        
        #tau, dtau, coords = self.network.out_grad(Xp)

        tau, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        
        #! Xe CHANGE: bounded slices
        D = Xp[:,self.dim:2*self.dim]-Xp[:,:self.dim]
        T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D))

        LogTau = torch.log(tau[:,0])
        LogTau2 = LogTau**2

        A = 2*LogTau*T0/tau[:,0]
        B = LogTau2/T0
        #print(A.shape)

        Ypred0 = -A*dtau[:,:self.dim]+B*D
        #print(Ypred0.shape)
        Spred0 = torch.norm(Ypred0)
        Ypred0 = 1/Spred0**2*Ypred0

        Ypred1 = -A*dtau[:,self.dim:2*self.dim]-B*D
        Spred1 = torch.norm(Ypred1)

        Ypred1 = 1/Spred1**2*Ypred1
        
        return torch.cat((Ypred0, Ypred1),dim=1)

    def TravelTimes(self, Xp):
        """Arrival time field:  T = log(tau)^2 * ||qs - qg||

        Xp may be (N, 2*dim) or (N, 3*dim); the trailing qe block is carried
        for interface consistency but is NOT multiplied into T.

        WHY NOT: the enemy slow-down is baked into the SPEED field the network
        is trained on (see apply_enemy_slowdown). Because T solves the Eikonal
        equation for that speed, T *already* reflects the slow-down: arrival
        times near Xe grow, since crossing that region genuinely costs more
        time. Multiplying an extra factor in here on top would double-count it
        and, worse, would invalidate the hand-derived derivatives in
        Loss/Gradient/Speed, which assume exactly this functional form.
        """
        # Apply projection from LatLong to UTM
        Xp = Xp.to(torch.device(self.Params['Device']))
        #! Xe CHANGE: accept 2*dim or 3*dim input
        Xp = self.append_enemy(Xp)
        
        tau, coords = self.network.out(Xp)

       
        D = Xp[:,self.dim:2*self.dim]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = torch.log(tau[:, 0])**2* torch.sqrt(T0)

        del Xp, tau, T0
        return TT
    
    def Tau(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
        #! Xe CHANGE: accept 2*dim or 3*dim input
        Xp = self.append_enemy(Xp)
     
        tau, coords = self.network.out(Xp)

        return tau

    def Speed(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
        #! Xe CHANGE: accept 2*dim or 3*dim input
        Xp = self.append_enemy(Xp)

        tau, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        #Xp.requires_grad_()
        #tau, dtau, coords = self.network.out_grad(Xp)

        
        #! Xe CHANGE: bounded slices
        D = Xp[:,self.dim:2*self.dim]-Xp[:,:self.dim]
        T0 = torch.einsum('ij,ij->i', D, D)

        DT1 = dtau[:,self.dim:2*self.dim]

        T3    = tau[:,0]**2

        LogTau = torch.log(tau[:,0])
        LogTau2 = LogTau**2
        LogTau3 = LogTau**3
        LogTau4 = LogTau**4

        T1    = 4*LogTau2*T0/T3*torch.einsum('ij,ij->i', DT1, DT1)
        T2    = 4*LogTau3/tau[:,0]*torch.einsum('ij,ij->i', DT1, D)

        
        
        S = (T1+T2+LogTau4)

        Ypred = 1/torch.sqrt(S)
        
        del Xp, tau, dtau, T0, T1, T2, T3, LogTau,LogTau2,LogTau3,LogTau4,DT1,S
        return Ypred


    def plot(self, src, tar, epoch, total_train_loss, alpha, cur_points=None, camera_matrix=None, traj_list = None):
        limit = 1
        xmin = [-0.5, -0.5]
        xmax = [0.5, 0.5]
        spacing=limit/80.0
        X,Y      = np.meshgrid(np.arange(xmin[0],xmax[0],spacing),np.arange(xmin[1],xmax[1],spacing))

        Xsrc = src 
        # Xsrc[2] = 0.2
        # Xsrc = [0.2, -0.2, 0]
        
        #! Xe CHANGE: XP is now 3*dim -> (qs | qg | qe)
        XP       = np.zeros((len(X.flatten()),3*self.dim))#*((xmax[dims_n]-xmin[dims_n])/2 +xmin[dims_n])
        XP[:,:self.dim] = Xsrc
        XP[:,self.dim:2*self.dim] = Xsrc
        #XP=XP/scale
        XP[:,self.dim+0]  = X.flatten()
        XP[:,self.dim+1]  = Y.flatten() #! this allow to change from y to z
        #! Xe CHANGE: qe block (Xe is always valid)
        XP[:,2*self.dim:3*self.dim] = self.Xe
        XP = Variable(Tensor(XP)).to(self.Params['Device'])
        
        tt = self.TravelTimes(XP)
        ss = self.Speed(XP)#*5
        tau = self.Tau(XP)
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        V  = ss.to('cpu').data.numpy().reshape(X.shape)
        TAU = tau.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        # ax.invert_xaxis()
        # ax.invert_yaxis()
        quad1 = ax.pcolormesh(X,Y,V,vmin=0,vmax=1)


        #! camera triangle
        if camera_matrix is not None:
            orientation_matrix = camera_matrix[:3, :3]  # Assuming the rotation part is the 3x3 upper-left submatrix
            position = src

            # Calculate the orientation angle from the orientation matrix
            yaw = np.arctan2(orientation_matrix[1, 0], orientation_matrix[0, 0])  # You can use this angle to represent the camera look at
            yaw += np.pi/2

            # Create a small triangle marker for the camera
            triangle_size = 0.2  # Adjust the size as needed
            triangle_marker = np.array([[0, 0], [triangle_size/2, -triangle_size/2], [triangle_size / 2, triangle_size/2]])
            
            # Rotate the triangle marker to match the camera's orientation
            rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rotated_triangle_marker = np.dot(triangle_marker, rotation_matrix.T)

            # # Add the triangle marker at the camera position
            # ax.plot(rotated_triangle_marker[0], rotated_triangle_marker[1], 'r^', markersize=10)

            camera_x = position[0] + rotated_triangle_marker[:, 0]
            camera_y = position[1] + rotated_triangle_marker[:, 1]
            ax.fill(camera_x, camera_y, 'b')

    
        #! plot trajectory
        if traj_list is not None:
            ax.plot(traj_list[:, 0], traj_list[:, 1], color='pink', marker = 'o', markersize=0.8, linestyle='-')
            traj_list_path = self.folder+"/"+str(epoch)+".npy"
            # np.save(traj_list_path, traj_list)
            plt.savefig(self.folder+"/plots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.png",bbox_inches='tight')

            #? plot trajectory with step size
            ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], color='red', marker='o', markersize=0.8, linestyle='-', linewidth=1)
            

        #! Xe CHANGE: draw the enemy/target and its slow-down zone
        if True:
            ax.plot(self.Xe[0], self.Xe[1], marker='*', color='red',
                    markersize=14, markeredgecolor='white', markeredgewidth=0.8,
                    zorder=5, linestyle='None')
            ax.add_patch(plt.Circle((self.Xe[0], self.Xe[1]), self.enemy_d_slow,
                                    fill=False, color='red', linestyle='--',
                                    linewidth=0.8, zorder=5))

        ax.contour(X,Y,TT,np.arange(0,5,0.02), cmap='bone', linewidths=0.3)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.folder+"/plots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.png",bbox_inches='tight')

        plt.close(fig)

        # draw occupancy grid
        if self.mode in [EXPLORATION]:
            pathname = self.folder+"/plots"+str(epoch)+"_"+"occ"+".png"
            cur_small_loc = src.cpu().numpy()
            tar_small_loc = tar.cpu().numpy()
            self.occ_map.plot(cur_small_loc, tar_small_loc, path=pathname)
            

        if cur_points is not None:
            # print("plot: cur point shape:",cur_points.shape)
            start_points = cur_points[:,:3]
            end_points = cur_points[:,3:]
            diff = start_points - end_points
            dists = np.linalg.norm(diff, axis=-1)
            # print("dists:", dists.shape)
            plt.hist(dists, bins=100)
            plt.savefig(self.folder+"/plots_dist_"+str(epoch)+".png")
            plt.close()

    def predict_trajectory2(self, Xsrc, Xtar, step_size=0.03, tol=0.03,
                            max_iter=200, return_status=False):
        #! Xe CHANGE: `return_status` is opt-in so existing callers
        #! (e.g. policy_occ) keep receiving a bare trajectory array.
        #! When True, returns (traj, converged) where converged=True means the
        #! bidirectional descent actually met in the middle -> a collision-free
        #! route exists in the currently mapped space. False means it hit the
        #! iteration cap, which usually means the map does not connect yet.
        Xsrc = Tensor(Xsrc)
        Xtar = Tensor(Xtar)
        XP_traj= torch.cat((Xsrc,Xtar))
        dis= torch.norm(XP_traj[:self.dim]-XP_traj[self.dim:])
        point_start = []
        point_goal = []
        iter = 0
        while dis > tol:
            gradient = self.Gradient(XP_traj[None,:].clone())
            XP_traj = XP_traj + step_size * gradient[0].cpu()
            XP_traj[2] = Xsrc[2]
            XP_traj[5] = Xsrc[2]
            dis = torch.norm(XP_traj[3:6]-XP_traj[0:3])
            point_start.append(XP_traj[0:3])
            point_goal.append(XP_traj[3:6])
            iter = iter + 1
            if iter > max_iter:
                break 
        
        #! Xe CHANGE: did the two ends actually meet?
        converged = bool(dis <= tol)

        point_goal.reverse()
        point = point_start + point_goal
        if point_start:
            traj = torch.cat(point).view(-1, 3)
            traj = torch.cat((Xsrc[None,], traj, Xtar[None,]), dim=0)
        else:
            traj = torch.cat((Xsrc[None,], Xtar[None,]), dim=0)
        # traj = np.concatenate([Xsrc[None,], traj, Xtar[None,]], axis=0)

        traj = traj.detach().numpy()
        if return_status:
            return traj, converged
        return traj
    

    def predict_trajectory(self, Xsrc, Xtar, step_size=0.03, tol=0.02):
        Xsrc = Tensor(Xsrc).cuda()
        Xtar = Tensor(Xtar).cuda()
        XP_traj= torch.cat((Xsrc,Xtar))
        dis= torch.norm(XP_traj[:self.dim]-XP_traj[self.dim:])
        point_start = []
        point_goal = []
        iter = 0
        while dis > tol:
            gradient = self.Gradient(XP_traj[None,:].clone())
            XP_traj = XP_traj + step_size * gradient[0]#.cpu()
            dis = torch.norm(XP_traj[3:6]-XP_traj[0:3])
            point_start.append(XP_traj[0:3])
            point_goal.append(XP_traj[3:6])
            iter = iter + 1
            if iter > 100:
                break 
        
        point_goal.reverse()
        point = point_start + point_goal
        if point_start:
            traj = torch.cat(point).view(-1, 3)
            traj = torch.cat((Xsrc[None,], traj, Xtar[None,]), dim=0)
        else:
            traj = torch.cat((Xsrc[None,], Xtar[None,]), dim=0)
        # traj = np.concatenate([Xsrc[None,], traj, Xtar[None,]], axis=0)

        return traj.detach().cpu().numpy()
    
    def policy_occ(self, current_location, height):
        # curtargetblock = self.occ_map.get_target_block()
        # if not curtargetblock:
        #     return Variable(Tensor([0, 0, height])).cpu()
        # curtargetloc = self.occ_map.get_block_center(curtargetblock)
        possible_locs = self.occ_map.get_block_centers()
        if True:
            #? find the shortest path one
            #! Xe CHANGE / SEARCH PRIORITY: the objective is FINDING Xe, not
            #! maximising coverage. Each candidate frontier is scored by
            #!     score = T(robot -> frontier) + search_weight * T(frontier -> Xe)
            #! The first term keeps the choice reachable/cheap (the paper's
            #! behaviour); the second pulls the sweep toward the target.
            #! search_weight = 0 reproduces the original coverage sweep.

            #! (a) cost from the robot to each frontier
            XP = np.zeros((len(possible_locs),3*self.dim))
            XP[:,:self.dim] = current_location
            XP[:,self.dim:self.dim+2] = possible_locs
            XP[:,self.dim+2] = height
            XP[:,2*self.dim:3*self.dim] = self.Xe
            XP = Variable(Tensor(XP)).to(self.Params['Device'])
            tt_to_frontier = self.TravelTimes(XP).to('cpu').data.numpy()

            #! (b) cost from each frontier onward to Xe.
            #! Xe CHANGE / ZERO-WEIGHT FIX: when the weight is 0 we SKIP this
            #! entirely rather than computing it and multiplying by zero.
            #! TravelTimes can return inf for an unreachable frontier, and
            #! 0.0 * inf = NaN in IEEE arithmetic - which then wins argmin and
            #! selects a garbage frontier. Skipping is both correct and cheaper
            #! (it also avoids a whole forward pass).
            if self.search_weight != 0.0:
                XPe = np.zeros((len(possible_locs),3*self.dim))
                XPe[:,:self.dim] = XP[:,self.dim:2*self.dim].cpu().numpy()
                XPe[:,self.dim:2*self.dim] = self.Xe
                XPe[:,2*self.dim:3*self.dim] = self.Xe
                XPe = Variable(Tensor(XPe)).to(self.Params['Device'])
                tt_to_enemy = self.TravelTimes(XPe).to('cpu').data.numpy()
            else:
                tt_to_enemy = np.zeros(len(possible_locs))

            #! (c) predicted SPEED at each frontier - same network, same
            #! forward pass, no extra machinery. Speed() evaluates at the goal
            #! block, which is the frontier here.
            #! Xe CHANGE / ZERO-WEIGHT FIX: same reasoning - skip the Speed()
            #! forward+backward pass entirely when the weight is 0.
            if self.speed_weight != 0.0:
                s_frontier = self.Speed(XP).to('cpu').data.numpy()
            else:
                s_frontier = np.ones(len(possible_locs))
            #! Xe CHANGE / NaN FIX: Speed() is 1/sqrt(S) and S can reach 0 or
            #! go slightly negative early in training, giving inf/NaN. np.clip
            #! does NOT remove NaN, and np.argmin then RETURNS THE NaN INDEX -
            #! i.e. the robot silently picks an arbitrary frontier. Replace
            #! non-finite values before clipping.
            s_frontier = np.nan_to_num(s_frontier, nan=0.0,
                                       posinf=1.0, neginf=0.0)
            s_frontier = np.clip(s_frontier, 0.0, 1.0)

            #! Xe CHANGE / NaN FIX: the time terms can also be non-finite
            #! before the field has converged. Map them to +inf so they are
            #! never selected, rather than letting NaN win the argmin.
            n_bad = int((~np.isfinite(tt_to_frontier)).sum()
                        + (~np.isfinite(tt_to_enemy)).sum())
            if n_bad > 0:
                print("[Xe] WARNING: {} non-finite arrival times among {} "
                      "frontiers - excluded from selection".format(
                          n_bad, len(possible_locs)))
            big = 1e6
            tt_to_frontier = np.nan_to_num(tt_to_frontier, nan=big,
                                           posinf=big, neginf=big)
            tt_to_enemy = np.nan_to_num(tt_to_enemy, nan=big,
                                        posinf=big, neginf=big)

            #! (d) combined objective: low arrival time, low remaining time to
            #! Xe, and HIGH speed at the chosen viewpoint.
            score = (tt_to_frontier
                     + self.search_weight * tt_to_enemy
                     + self.speed_weight * (1.0 - s_frontier))
            small_idx = np.argmin(score)
            curtargetloc_small = possible_locs[small_idx]

            print("[Xe] policy_occ: {} frontiers | w_target={:.2f} w_speed={:.2f}".format(
                      len(possible_locs), self.search_weight, self.speed_weight))
            print("[Xe]   chosen #{}: T_frontier={:.4f}  T_to_Xe={:.4f}  "
                  "S_frontier={:.3f}  ->  score={:.4f}".format(
                      int(small_idx),
                      float(tt_to_frontier[small_idx]),
                      float(tt_to_enemy[small_idx]),
                      float(s_frontier[small_idx]),
                      float(score[small_idx])))
            print("[Xe]   pool: best T_to_Xe={:.4f}  fastest S={:.3f}  "
                  "slowest S={:.3f}".format(
                      float(tt_to_enemy.min()), float(s_frontier.max()),
                      float(s_frontier.min())))
        
        print("current location :", current_location)
        print("target location :", curtargetloc_small)
        print("block idx:", self.occ_map.block_stack[small_idx])

        #！ set a step size
        curtargetloc_small = np.concatenate([curtargetloc_small, [height]])
        curtargetloc_small_tensor = torch.from_numpy(curtargetloc_small).float()
        
        curloc_tensor = torch.from_numpy(current_location).float()

        traj_list = self.predict_trajectory2(curloc_tensor, curtargetloc_small_tensor, step_size=0.02)

        # find the largest point that is smaller than the step size
        if True:
            #? accumulate the distance to fit the step size
            step_size = 0.05
            index = 0
            curtargetloc = traj_list[-1]
            accum_dis = 0
            while accum_dis < step_size:
                curtargetloc = traj_list[index]
                accum_dis += np.linalg.norm(traj_list[index+1][:2] - traj_list[index][:2])
                index += 1
                if index >= len(traj_list)-1:
                    break
            print("accumulated distance:", accum_dis, "at traj_index:", index)
            print("Target location after gradient descent:", curtargetloc)
            print("Length of traj_list:", len(traj_list))

        #! should return a list of trajectory, and index of the end traj
        # return Variable(Tensor(curtargetloc)).cpu()
        return traj_list, index - 1