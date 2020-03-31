import math
import torch
import gpytorch
from poisson_process import PoissonProcess
from gaussian_process import ExactGPModel
from torch import nn

class CoxProcess(object):
    """docstring for CoxProcess."""

    def __init__(self, intensity_grid, log_intensity=None, observations=None,
                 **kwargs):
        super(CoxProcess, self).__init__()
        self.intensity_grid = intensity_grid
        if log_intensity is not None:
            self.log_intensity = nn.Parameter(log_intensity)
        else:
            self.log_intensity = nn.Parameter(torch.zeros(intensity_grid.size(0)))
        self.observations = observations

        self.gp_lh = gpytorch.likelihoods.GaussianLikelihood()
        self.poisson_process = PoissonProcess(self.intensity_grid, self.log_intensity,
                                                self.observations)
        self.gaussian_process = ExactGPModel(self.intensity_grid,
                                            self.log_intensity, self.gp_lh,
                                            **kwargs)

    def learn_intensity(self, observations=None,
                        iters=100):
        """
        newtons method to learn the intensity function
        """
        self.gp_lh.train()
        self.gaussian_process.train()
        K = self.gaussian_process.covar_module(self.intensity_grid).evaluate()
        n = K.size(0)
        K_inv = torch.inverse(K + torch.eye(n)*1e-4)
        if observations is not None:
            self.observations = observations

        # print(K_inv)
        self.gp_lh.eval()
        self.gaussian_process.eval()

        gp_mean = self.gaussian_process.mean_module(self.intensity_grid)
        # print(gp_mean)

        f_curr = self.log_intensity
        for iter in range(iters):
            self.poisson_process.update_val(f_curr)

            ## first order derivative ##
            pois_grad = self.poisson_process.compute_grad(self.observations)
            # print("Pois GRAD")
            # print(pois_grad)
            cox_grad = pois_grad  - K_inv.matmul(f_curr - gp_mean)
            # print("COX GRAD")
            # print(cox_grad)
            # print("\n\n")

            ## second order derivative ##
            pois_hess = self.poisson_process.compute_hessian(self.observations)
            # print(pois_hess)
            cox_hess = pois_hess - K_inv
            # print(cox_hess)

            f_curr = f_curr - (cox_hess).inverse().matmul(cox_grad)

        self.log_intensity = f_curr # update the log intensity to the inferred value

        return f_curr

    def stable_learn_intensity(self, observations=None,
                        iters=100):
        """
        section 6.1 of fast kronecker inference with non-gaussian likelihoods
        """
        self.gp_lh.train()
        self.gaussian_process.train()
        K = self.gaussian_process.covar_module(self.intensity_grid).evaluate()
        n = K.size(0)
        K_inv = torch.inverse(K + torch.eye(n)*1e-4)
        if observations is not None:
            self.observations = observations

        # print(K_inv)
        self.gp_lh.eval()
        self.gaussian_process.eval()

        gp_mean = self.gaussian_process.mean_module(self.intensity_grid)
        # print(gp_mean)

        f_curr = self.log_intensity
        for iter in range(iters):
            self.poisson_process.update_val(f_curr)
            pois_grad = self.poisson_process.compute_grad(self.observations)

            W = -1 * self.poisson_process.compute_hessian(self.observations)
            W_root = torch.diag(W.diag().pow(0.5))

            B = torch.eye(W.size(0)) + W_root.matmul(K_inv).matmul(W_root)

            Q = W_root.matmul( torch.inverse(B) ).matmul(W_root)

            b = W.matmul(f_curr - gp_mean) + pois_grad
            a = b - Q.matmul(K).matmul(b)

            f_curr = K.matmul(a)

        self.log_intensity = f_curr # update the log intensity to the inferred value

        return f_curr

    def learn_hypers(self, iters=200):
        self.gp_lh.train()
        self.gaussian_process.train()

        optimizer = torch.optim.Adam([
            {'params': self.gaussian_process.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_lh, self.gaussian_process)

        for i in range(iters):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.gaussian_process(self.intensity_grid.detach())
            # Calc loss and backprop gradients
            loss = -mll(output, self.log_intensity.detach())
            loss.backward()
            optimizer.step()
            print(loss.item())
