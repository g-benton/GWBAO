import torch
import math
import gpytorch
from gpytorch.kernels import RBFKernel

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=RBFKernel(),
                scale=True):
        if train_y is None:
            train_y = torch.zeros(train_x.size(0))
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if scale:
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
        else:
            self.covar_module = kernel


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
