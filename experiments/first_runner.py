import math
import torch
import gpytorch
import pyro
import tqdm
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append("../")
from kernels import SpaceKernel
from torch.nn.functional import softplus


class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_arrivals, edge_len, inducing_pts, name_prefix="cox_gp_model"):
        self.name_prefix = name_prefix
        self.dim = inducing_pts.shape[-1]
        self.edge_len = edge_len
        self.mean_intensity = num_arrivals / (edge_len ** dim)
        num_inducing = inducing_pts.shape[0]

        # Define the variational distribution and strategy of the GP
        # We will initialize the inducing points to lie on a grid from 0 to T
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_pts, variational_distribution)

        # Define model
        super().__init__(variational_strategy=variational_strategy)

        # Define mean and kernel
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = SpaceKernel()

    def forward(self, times):
        mean = self.mean_module(times)
        covar = self.covar_module(times)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, arrival_times, quadrature_times):
        # Draw samples from q(f) at arrival_times
        # Also draw samples from q(f) at evenly-spaced points (quadrature_times)
        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            pyro.sample(
                self.name_prefix + ".function_samples",
                self.pyro_guide(torch.cat([arrival_times, quadrature_times], 0))
            )

    def model(self, arrival_times, quadrature_times):
        pyro.module(self.name_prefix + ".gp", self)

        # Draw samples from p(f) at arrival times
        # Also draw samples from p(f) at evenly-spaced points (quadrature_times)
        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            function_samples = pyro.sample(
                self.name_prefix + ".function_samples",
                self.pyro_model(torch.cat([arrival_times, quadrature_times], 0))
            )

        ####
        # Convert function samples into intensity samples, using the function above
        ####
        intensity_samples = function_samples.exp() * self.mean_intensity

        # Divide the intensity samples into arrival_intensity_samples and quadrature_intensity_samples
        arrival_intensity_samples, quadrature_intensity_samples = intensity_samples.split([
            arrival_times.size(0), quadrature_times.size(0)
        ], dim=-1)

        ####
        # Compute the log_likelihood, using the method described above
        ####
        arrival_log_intensities = arrival_intensity_samples.log().sum(dim=-1)

        ## avg intensity
        est_num_arrivals = quadrature_intensity_samples.mean(dim=-1).mul(self.edge_len ** self.dim)
        log_likelihood = arrival_log_intensities - est_num_arrivals
        pyro.factor(self.name_prefix + ".log_likelihood", log_likelihood)

def main():
    torch.random.manual_seed(99)

    ## fixed parameters for generation ##
    r_bao = torch.tensor(100/0.7/1000) # Gpc
    w_bao = torch.tensor(15/0.7/1000) # Gpc
    raw_r_bao = torch.log(torch.exp(r_bao) - 1)
    raw_w_bao = torch.log(torch.exp(w_bao) - 1)

    ## overwrite kernel to have correct parameters ##
    kern = SpaceKernel()
    kern.raw_gauss_mean.data = raw_r_bao
    kern.raw_gauss_sig.data = raw_w_bao

    ## load in observations ##
    f = h5py.File("../data/comoving-positions.h5", 'r')
    dset = f['pos']
    obs = torch.FloatTensor(dset[()])
    num_obs = 1746 ## just taken from sample dataset

    ###############################
    ## MODEL AND INDUCING POINTS ##
    ###############################

    ## generate inducing points ##
    n = 10
    dim = 3
    inducing_pts = torch.zeros(pow(n, dim), dim)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                inducing_pts[i * n**2 + j * n + k][0] = float(i) / ((n-1) * 0.5) - 1.
                inducing_pts[i * n**2 + j * n + k][1] = float(j) / ((n-1) * 0.5) - 1.
                inducing_pts[i * n**2 + j * n + k][2] = float(k) / ((n-1) * 0.5) - 1.

    inducing_row = torch.tensor([float(i) / ((n-1) * 0.5) - 1. for i in range(n)])

    inducing_pts = inducing_pts.float()

    ## set up model ##
    model = GPModel(obs.shape[0], edge_len = 2., inducing_pts=inducing_pts)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        inducing_pts = inducing_pts.cuda()
        model = model.cuda()

    ###################
    ## GENERATE DATA ##
    ###################
    Ndraw = 8000
    rs = np.cbrt(0.74**3*torch.rand(Ndraw).numpy())

    cos_thetas = np.random.uniform(low=-1, high=1, size=Ndraw)
    sin_thetas = np.sqrt(1-cos_thetas*cos_thetas)
    phis = np.random.uniform(low=0, high=2*math.pi, size=Ndraw)

    pts = np.column_stack((rs*np.cos(phis)*sin_thetas,
                    rs*np.sin(phis)*sin_thetas,
                    rs*cos_thetas))

    rs = np.sqrt(np.sum(np.square(pts[:,np.newaxis,:] - pts[np.newaxis,:,:]), axis=2))

    sample_intensity = model(torch.tensor(pts).float()).sample(sample_shape=torch.Size((1,))).squeeze()
    sample_intensity = sample_intensity.div(sample_intensity.max())
    pts = pts[torch.rand(Ndraw) < sample_intensity, :]
    print('Drew {:d}'.format(pts.shape[0]))

    pts = pd.DataFrame(data=pts, columns=['x', 'y', 'z'])

    ################################
    ## NOW TRAIN A NEW COVARIANCE ##
    ################################
    model.covar_module = SpaceKernel()

    import os
    num_iter = 250
    num_particles = 32

    train_pts = torch.tensor(pts.values).float()
    inducing_pts = inducing_pts.float()

    def train(lr=0.01):
        optimizer = pyro.optim.Adam({"lr": lr})
        loss = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=False, retain_graph=True)
    #     loss = pyro.infer.Trace_ELBO()
        infer = pyro.infer.SVI(model.model, model.guide, optimizer, loss=loss)

        model.train()
        for i in range(num_iter):
            loss = infer.step(train_pts, inducing_pts)
            loader.set_postfix(loss=loss)

            if i % 25 == 0:
                print("iter = ", i)

    train()


    function_dist = model(inducing_pts)
    intensity_samples = function_dist(torch.Size([1000])).exp() * model.mean_intensity

    fpath = "./saved_gwbao_model.pt"
    torch.save(model.state_dict(), fpath)

    ##############
    ## PLOTTING ##
    ##############

    base_kern = SpaceKernel()
    true_kern = SpaceKernel()
    true_kern.raw_gauss_mean.data = raw_r_bao
    true_kern.raw_gauss_sig.data = raw_w_bao

    learn_cov = model.covar_module(tau, torch.zeros(1,1)).evaluate()
    base_cov = base_kern(tau, torch.zeros(1,1)).evaluate()
    true_cov = true_kern(tau, torch.zeros(1,1)).evaluate()

    plt.plot(tau, learn_cov.detach(), label="learned")
    plt.plot(tau, base_cov.detach(), label="initial")
    plt.plot(tau, true_cov.detach(), label="truth")
    plt.yscale("log")
    plt.legend()
    plt.title("(Log) GP Covariance for Cox Process")
    plt.xlabel(r"$\tau$")
    plt.ylabel("Log Covariance")

    plt.savefig("./learned_covariance.pdf", bbox_inches="tight")

    
if __name__ == '__main__':
    main()
