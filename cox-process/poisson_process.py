import math
import torch
import numpy as np
from torch.distributions import Poisson
from torch import nn
from torch import autograd

class PoissonProcess(object):
    """docstring for PoissonProcess"""

    def __init__(self, intensity_grid, log_intensity=None, observations=None):
        super(PoissonProcess, self).__init__()

        self.intensity_grid = intensity_grid
        if log_intensity is not None:
            self.log_intensity = nn.Parameter(log_intensity)
        else:
            self.log_intensity = nn.Parameter(torch.zeros(intensity_grid.size(0)))
        self.observations = observations
        self.in_dim = self.intensity_grid.shape[-1]
        self.n_pts = self.intensity_grid.shape[0]
        self.delta_x = torch.abs(self.intensity_grid[0, -1] - self.intensity_grid[1, -1])

    def simulate(self, intensity_grid=None, log_intensity=None):

        if intensity_grid is not None:
            self.update_grid(intensity_grid)
        if log_intensity is not None:
            self.update_val(log_intensity)

        sim_points = torch.tensor([])

        for pt in range(self.n_pts):
            ## draw points from poisson ##
            rate = torch.exp(self.log_intensity[pt]) * (self.delta_x ** self.in_dim)
            dist = Poisson(rate)
            n_draw = int(dist.sample().item())

            samples = torch.zeros(n_draw, self.in_dim)
            ## sample their locations ##
            for dim in range(self.in_dim):
                samples[:, dim] = self.delta_x * torch.rand(n_draw)
                samples[:, dim] += self.intensity_grid[pt, dim] + self.delta_x/2

            ## append to sim_points ##
            sim_points = torch.cat((sim_points, samples))

        return sim_points

    def update_grid(self, grid):
        self.intensity_grid = grid

    def update_val(self, val):
        self.log_intensity.data = val

    def update_obs(self, obs):
        self.observations = obs

    def compute_obs_distance(self, observations=None):
        n = observations.size(0)
        m = self.intensity_grid.size(0)
        d = observations.size(1)

        xx = observations.unsqueeze(1).expand(n, m, d)
        yy = self.intensity_grid.unsqueeze(0).expand(n, m, d)

        dist = torch.pow(xx - yy, 2).sum(2)
        return dist

    def likelihood(self, observations=None):

        if observations is not None:
            self.update_obs(observations)

        dist = Poisson(self.delta_x.pow(self.in_dim) * self.log_intensity.exp())

        if type(observations) is list:
            ## if we're storing multiple draws ##
            lh = 0
            for obs in observations:
                obs_dist = self.compute_obs_distance(obs)
                samples_from = obs_dist.min(1)[1]

                counts_per_bin = torch.zeros(self.intensity_grid.size(0))
                for smp in samples_from:
                    counts_per_bin[smp] += 1

                lh += dist.log_prob(counts_per_bin).sum()
            return lh
        else:
            ## storing a single draw
            obs_dist = self.compute_obs_distance(observations)
            samples_from = obs_dist.min(1)[1]

            counts_per_bin = torch.zeros(self.intensity_grid.size(0))
            for smp in samples_from:
                counts_per_bin[smp] += 1

            lh = dist.log_prob(counts_per_bin).sum()
            return lh

    def compute_grad(self, observations=None):
        if observations is not None:
            self.update_obs(observations)
        lh = self.likelihood(observations)
        lh.backward()
        return self.log_intensity.grad

    def compute_hessian(self, observations=None):
        if observations is not None:
            self.update_obs(observations)
        lh = self.likelihood(observations)
        grad = autograd.grad(lh, self.log_intensity, create_graph=True)[0]
        grad = autograd.grad(grad.sum(), self.log_intensity)[0]
        return torch.diag(grad)
