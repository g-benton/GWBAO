import math
import torch
import gpytorch
import copy
from gpytorch.kernels.kernel import Kernel
#from botorch import fit_gpytorch_model
from torch.nn import ModuleList
from torch.nn.functional import softplus


class SpaceKernel(Kernel):
    def __init__(self, A_scale=-1.8, **kwargs):
        super(SpaceKernel, self).__init__(**kwargs)

        # self.register_parameter(
        #     name="raw_lengthscale", parameter=torch.nn.Parameter(torch.zeros(1))
        # )

        self.register_parameter(
            name="raw_gauss_mean", parameter=torch.nn.Parameter(torch.ones(1,1)*-2.5)
        )

        self.register_parameter(
            name="raw_gauss_sig", parameter=torch.nn.Parameter(torch.ones(1, 1)*-3.5)
        )

        self.A_scale = A_scale
        self.cutoff_scale = 0.00714
        self.A = 0.13114754
        self.B = 0.01



    def forward(self, x1, x2=None, diag=False, last_dim_is_batch=False,
        **kwargs):

        x1_ = x1
        x2_ = x1 if x2 is None else x2
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
            x2_ = x2_.transpose(-1, -2).unsqueeze(-1)
            tau = x1_ - x2_.transpose(-2, -1)
        else:
            tau = self.covar_dist(x1, x2, square_dist=False, diag=False,
                                  last_dim_is_batch=last_dim_is_batch)

        gauss_mean = softplus(self.raw_gauss_mean)
        gauss_sig = softplus(self.raw_gauss_sig)

        # print(gauss_mean)

        pl_term = torch.where(tau < self.cutoff_scale,
                              1 + self.A_scale * (tau - self.cutoff_scale).div(self.cutoff_scale),
                              (tau.div(self.cutoff_scale)).pow(self.A_scale) * self.A)
        # pl_term = tau.div(self.cutoff_scale).pow(self.A_scale) * self.A

        bao_dist = torch.sqrt(2 * math.pi * gauss_sig).pow(-1)
        bao_dist = bao_dist * torch.exp(-1./2. * (tau - gauss_mean).div(gauss_sig).pow(2))

        # print(pl_term)
        # print(bao_dist)

        output = self.A * pl_term + self.B * bao_dist

        if diag:
            output = output.diag()
        return output
