import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from lidarmodel import p_z_Lidar_prior, p_z_Lidar_posterior, p_z_Map_prior, lidar_generate_obs_labels
from constraints import P_z_C1, P_z_C2
from barrierfunctions import sigmoidSS


from sys import path
from os.path import dirname as dir
path.append(dir(path[0].replace("/robotexploration", "")))
__package__ = "probmind"

from Planning import Planning

# matplotlib.use("TkAgg")

class RobotExploration(Planning):

    def __init__(self, meter2pixel, lidar_range, lidar_FOV, lidar_resolution, lidar_sigma_hit, d_min):
        # K = 1
        K = 1  # K: number of options/trajectories to consider
        M = 2  # M: number of samples from each independent perception in calculation of the information gain
        N = 3  # N: number of LTM samples used in calculation of the information gain
        G = 1  # G: number of samples from each independent constraint

        self.N_posterior_samples = 30

        desirability_scale_factor = 0.1
        # progress_scale_factor = 0.01
        # info_gain_scale_factor = 0.0075
        progress_scale_factor = 0.01
        info_gain_scale_factor = 2.0
        svi_epochs = 30

        # https://pyro.ai/examples/svi_part_iv.html
        initial_lr = 0.05
        gamma = 0.2  # final learning rate will be gamma * initial_lr
        # initial_lr = 0.1
        # gamma = 0.1  # final learning rate will be gamma * initial_lr
        lrd = gamma ** (1 / svi_epochs)
        optim_args = {'lr': initial_lr, 'lrd': lrd}
        optimizer = pyro.optim.ClippedAdam(optim_args)

        # Model specific params:
        standard_diviation = torch.tensor(0.2 / 3)  # 99.7% of samples within a circle of 25 cm
        variance = standard_diviation * standard_diviation
        self.params["cov_s"] = variance * torch.eye(2)
        self.params["a_support"] = torch.tensor([2, 2], dtype=torch.float)  # 2 m in each direction

        # lidar params
        lidarParams = {}
        lidarParams["meter2pixel"] = meter2pixel
        lidarParams["z_max"] = torch.tensor([lidar_range / meter2pixel], dtype=torch.float)  # range in meters
        lidarParams["sigma_hit"] = torch.tensor(lidar_sigma_hit, dtype=torch.float)
        lidarParams["lambda_short"] = None  # not implemented
        lidarParams["N_lidar_beams"] = int(lidar_FOV / lidar_resolution)
        # lidarParams["N_lidar_beams_samples"] = 3
        lidarParams["N_lidar_beams_samples"] = 1
        lidarParams["P_hit"] = 0.99999
        lidarParams["P_rand"] = 0.00001
        lidarParams["P_max"] = 0.0
        lidarParams["P_short"] = 0.0  # not implemented yet! => should always be zero

        # it might be necessary to include a "lidarSubsampler" class instance when doing parallel simulations?
        self.params["lidarParams"] = lidarParams

        # constraint params
        # d_min = 0.15
        self.params["P_z_C1_scale"] = torch.tensor([15], dtype=torch.float)
        self.params["x_min"] = torch.tensor([0.0], dtype=torch.float)
        self.params["x_max"] = torch.tensor([2 * d_min], dtype=torch.float)
        self.params["P_z_C2_scale"] = 50
        self.params["lidarParams_constraints"] = lidarParams.copy()
        # self.params["lidarParams_constraints"]["N_lidar_beams_samples"] = 8
        self.params["lidarParams_constraints"]["N_lidar_beams_samples"] = 16

        super().__init__(K,
                         M,
                         N,
                         G,
                         svi_epochs,
                         optimizer,
                         desirability_scale_factor=desirability_scale_factor,
                         progress_scale_factor=progress_scale_factor,
                         info_gain_scale_factor=info_gain_scale_factor)

    def makePlan(self, t, T_delta, p_z_s_t, map_grid_probabilities, return_mode="mean", show_attention_map=False, p_z_g=None):
        LTM = {}
        LTM["map_grid_probabilities"] = map_grid_probabilities
        z_a_tauPlus_samples, z_s_tauPlus_samples = super().makePlan(t, T_delta, p_z_s_t, LTM, N_posterior_samples=self.N_posterior_samples, p_z_g=p_z_g)

        if not (return_mode == "mean" or return_mode == "raw_samples" or return_mode == "random"):
            return_mode == "random"

        if return_mode == "mean":
            # calculate the mean of the samples drawn from the posterior distribution:
            z_a_tauPlus_mean, z_s_tauPlus_mean = self.calculate_mean_state_action_means(z_a_tauPlus_samples, z_s_tauPlus_samples)

            z_a_tauPlus = z_a_tauPlus_mean
            z_s_tauPlus = z_s_tauPlus_mean
            for i in range(len(z_a_tauPlus)):
                z_a_tauPlus[i] = self.action_transforme(z_a_tauPlus[i]).detach()

        elif return_mode == "raw_samples":  # return random sample
            z_a_tauPlus = z_a_tauPlus_samples
            z_s_tauPlus = z_s_tauPlus_samples
            for j in range(self.N_posterior_samples):
                for i in range(len(z_a_tauPlus[0])):
                    z_a_tauPlus[j][i] = self.action_transforme(z_a_tauPlus[j][i]).detach()

        elif return_mode == "random":  # return random sample
            z_a_tauPlus = z_a_tauPlus_samples[0]
            z_s_tauPlus = z_s_tauPlus_samples[0]
            for i in range(len(z_a_tauPlus)):
                z_a_tauPlus[i] = self.action_transforme(z_a_tauPlus[i]).detach()

        if show_attention_map:
            self.attention_map(t, p_z_s_t, z_s_tauPlus)

        return z_a_tauPlus, z_s_tauPlus

    def calculate_mean_state_action_means(self, z_a_tauPlus_samples, z_s_tauPlus_samples):

        z_a_tauPlus_mean = []
        for tau in range(len(z_a_tauPlus_samples[0])):
            z_a_tau = torch.zeros_like(z_a_tauPlus_samples[0][tau])
            for i in range(self.N_posterior_samples):
                z_a_tau = z_a_tau + z_a_tauPlus_samples[i][tau]
            z_a_tauPlus_mean.append(z_a_tau / self.N_posterior_samples)

        z_s_tauPlus_mean = []
        for tau in range(len(z_s_tauPlus_samples[0])):
            z_s_tau = torch.zeros_like(z_s_tauPlus_samples[0][tau])
            for i in range(self.N_posterior_samples):
                z_s_tau = z_s_tau + z_s_tauPlus_samples[i][tau]
            z_s_tauPlus_mean.append(z_s_tau / self.N_posterior_samples)

        return z_a_tauPlus_mean, z_s_tauPlus_mean

    def attention_map(self, t, p_z_s_t, z_s_tauPlus, steps=10, area_size=2.0):
        # NOT UPDATED YET!!!
        z_s_t = p_z_s_t.mean

        x = np.arange(z_s_t[0] - area_size, z_s_t[0] + area_size, 2.0 / steps)
        y = np.arange(z_s_t[1] - area_size, z_s_t[1] + area_size, 2.0 / steps)
        xx, yy = np.meshgrid(x, y, sparse=False)
        shape = xx.shape

        P_z_d = np.zeros(shape)
        P_z_p = np.ones(shape)
        P_z_i = np.zeros(shape)
        P_z_c = np.ones(shape)
        P_z_A = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                z_s_tPlus1 = torch.tensor([xx[i, j], yy[i, j]], dtype=torch.float)
                P_z_s_tPlus1 = dist.MultivariateNormal(z_s_tPlus1, self.params["cov_s"])

                # calculate desirability
                if self.p_z_g is not None:
                    P_z_d[i, j] = self._Planning__P_z_d_tau(z_s_tPlus1, P_z_s_tPlus1, self.p_z_g)
                # calculate progress
                P_z_p[i, j] = self._Planning__P_z_p_tau(z_s_tPlus1, P_z_s_tPlus1, self.p_z_s_Minus)
                # calculate information gain
                P_z_i[i, j] = self._Planning__P_z_i_tau(z_s_tPlus1)
                # calculate constraints
                P_z_c[i, j] = self._Planning__P_z_c_tau(z_s_tPlus1, z_s_t)
                # calculate attention
                P_z_A[i, j] = self._Planning__P_z_A_tau(P_z_d[i, j], P_z_p[i, j], P_z_i[i, j], P_z_c[i, j])

        fig, axs = plt.subplots(1, 5)
        fig_title = "Attention Map for t = " + str(t)
        fig.suptitle(fig_title)
        # fig.canvas.manager.window.geometry("2500x600+0+700")
        mapShape = self.LTM["map_grid_probabilities"].shape
        axs[0].imshow(self.LTM["map_grid_probabilities"], cmap='binary', origin="upper", extent=[0, mapShape[1] / self.params["lidarParams"]["meter2pixel"], 0, mapShape[0] / self.params["lidarParams"]["meter2pixel"]], aspect="auto")
        cs0 = axs[0].contourf(x, y, P_z_d, alpha=0.75, vmin=0., vmax=1.)
        axs[0].set_title('P_z_d')

        axs[1].imshow(self.LTM["map_grid_probabilities"], cmap='binary', origin="upper", extent=[0, mapShape[1] / self.params["lidarParams"]["meter2pixel"], 0, mapShape[0] / self.params["lidarParams"]["meter2pixel"]], aspect="auto")
        cs0 = axs[1].contourf(x, y, P_z_p, alpha=0.75, vmin=0., vmax=1.)
        axs[1].set_title('P_z_p')

        axs[2].imshow(self.LTM["map_grid_probabilities"], cmap='binary', origin="upper", extent=[0, mapShape[1] / self.params["lidarParams"]["meter2pixel"], 0, mapShape[0] / self.params["lidarParams"]["meter2pixel"]], aspect="auto")
        cs0 = axs[2].contourf(x, y, P_z_i, alpha=0.75, vmin=0., vmax=1.)
        axs[2].set_title('P_z_i')

        axs[3].imshow(self.LTM["map_grid_probabilities"], cmap='binary', origin="upper", extent=[0, mapShape[1] / self.params["lidarParams"]["meter2pixel"], 0, mapShape[0] / self.params["lidarParams"]["meter2pixel"]], aspect="auto")
        cs0 = axs[3].contourf(x, y, P_z_c, alpha=0.75, vmin=0., vmax=1.)
        axs[3].set_title('P_z_c')

        axs[4].imshow(self.LTM["map_grid_probabilities"], cmap='binary', origin="upper", extent=[0, mapShape[1] / self.params["lidarParams"]["meter2pixel"], 0, mapShape[0] / self.params["lidarParams"]["meter2pixel"]], aspect="auto")
        cs0 = axs[4].contourf(x, y, P_z_A, alpha=0.75, vmin=0., vmax=1.)
        axs[4].set_title('P_z_A')
        fig.colorbar(cs0, use_gridspec=True)

        # draw planned trajectory
        for tau in range(len(z_s_tauPlus)):
            if tau == 0:
                z_s_tPlus = z_s_tauPlus[tau].detach().cpu().numpy()
            else:
                z_s_tPlus = np.vstack((z_s_tPlus, z_s_tauPlus[tau].detach().cpu().numpy()))
        axs[0].plot(z_s_tPlus[:, 0], z_s_tPlus[:, 1], color="red")
        axs[1].plot(z_s_tPlus[:, 0], z_s_tPlus[:, 1], color="red")
        axs[2].plot(z_s_tPlus[:, 0], z_s_tPlus[:, 1], color="red")
        axs[3].plot(z_s_tPlus[:, 0], z_s_tPlus[:, 1], color="red")
        axs[4].plot(z_s_tPlus[:, 0], z_s_tPlus[:, 1], color="red")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(.00001)

    # ################### Abstract methods of the class Planning ####################
    def q_z_a_tau(self, z_s_tauMinus1, k):
        alpha_init = torch.tensor([[10000., 10000.], [10000., 10000.]], dtype=torch.float)
        beta_init = torch.tensor([[10000., 10000.], [10000., 10000.]], dtype=torch.float)
        a_alpha = pyro.param("a_alpha_{}".format(k), alpha_init[k], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!
        a_beta = pyro.param("a_beta_{}".format(k), beta_init[k], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!

        _q_z_a_tau = dist.Beta(a_alpha, a_beta).to_event(1)
        z_a = pyro.sample("z_a", _q_z_a_tau)
        return z_a

    def p_z_a_tau(self, z_s_tau):
        # values should not be changed - use the params in "action_transforme" instead!
        a_min = torch.tensor([0., 0.], dtype=torch.float)
        a_max = torch.tensor([1., 1.], dtype=torch.float)
        _p_z_a_tau = dist.Uniform(a_min, a_max).to_event(1)
        z_a = pyro.sample("z_a", _p_z_a_tau)
        return z_a

    def p_z_s_tau(self, z_s_tauMinus1, z_a_tauMinus1):
        mean = z_s_tauMinus1 + self.action_transforme(z_a_tauMinus1)
        cov = self.params["cov_s"]
        _p_z_s_tau = dist.MultivariateNormal(mean, cov)
        z_s = pyro.sample("z_s", _p_z_s_tau)
        return z_s

    # def P_z_c_tau(self, z_s_tau, z_s_tauMinus1):
    #     # ################### SOMETHING NEEDS TO BE DONE WITH THE CONSTRAINTS!!! ###################
    #     # params = {}
    #     # params["lidarParams"] = self.params["lidarParams"]
    #     self.params["map"] = self.LTM["map_grid_probabilities"]

    #     _P_z_C1 = P_z_C1(z_s_tau, self.params)
    #     return [_P_z_C1]
    #     # _P_z_C2 = P_z_C2(z_s_tau, z_s_tauMinus1, self.params)
    #     # return [_P_z_C1, _P_z_C2]

    def I_c_tau(self, z_s_tau, z_s_tauMinus1):
        # d_min = torch.tensor(0.5)
        position = z_s_tau  # z_s_tau["position"]
        map_grid_probabilities = self.LTM["map_grid_probabilities"]
        lidar_generate_obs_labels(self.params["lidarParams_constraints"])  # to resamble beam angles...
        z_Map = p_z_Map_prior(position, map_grid_probabilities, self.params["lidarParams_constraints"])

        I_c = []
        for key in z_Map:
            if z_Map[key] is None:
                I_c.append(torch.tensor(1.0))
            else:
                dist = z_Map[key]
                I_c.append(1 - sigmoidSS(dist, self.params["P_z_C1_scale"], self.params["x_min"], self.params["x_max"]))
                # if z_Map[key] > d_min:
                #     I_c.append(torch.tensor(1.0))
                # else:
                #     I_c.append(torch.tensor(0.0))

        return I_c

    def p_z_LTM(self, z_s_tau):
        position = z_s_tau  # z_s_tau["position"]
        map_grid_probabilities = self.LTM["map_grid_probabilities"]
        z_Map = p_z_Map_prior(position, map_grid_probabilities, self.params["lidarParams"])
        z_LTM = {}
        z_LTM["z_Map"] = z_Map
        return z_LTM

    def p_z_PB(self, z_s_tau):
        p_z_Lidar_prior(self.params["lidarParams"])

    def p_z_PB_posterior(self, z_s_tau, z_LTM):
        z_Map = z_LTM["z_Map"]
        p_z_Lidar_posterior(z_Map, self.params["lidarParams"])

    def generate_obs_labels(self):
        observation_labels = lidar_generate_obs_labels(self.params["lidarParams"])
        return observation_labels

    # ################### other methods ####################
    def action_transforme(self, z_a):
        # scaling parameters for the action
        a_offset = -self.params["a_support"] / torch.tensor([2.], dtype=torch.float)

        return self.params["a_support"] * z_a + a_offset


# ################### TODO ###################
# 1) consider the relation between goal-states and reward functions - can they be modelled as the same thing?
# 6) consider noise on state in the simulator
# 8) Change state to a dict - e.g. z_s_tau["position"]
# 10) introduce a baseline when using goals
# 12) consider saving losses
# 13) implement rotation and FOV in lidar model
# 15) Explain the ".detach()" in "Lautum_information_estimate(...)"
# 16) implement constraint between states - the old "P_z_C2(...)" function

# ################### DONE? ###################

# ################### NEW IDEAS ###################
# 1) Introduce memorable states - i.e. states where we had multiple choices - this would also require the implementation of sub-goals
# 2) Implement hierarchical or parallel decisions/planning with intertwined subgoals and shared states, constraints, PB, etc. - via msg-passing algorithms?
# 3) Attention mechanism to direct sampling - i.e. we should consider more samples in the direction of the action
# 4) how to detect when an impasse occurs?
# 5) Add uncertainty to LTM variables that have not been updated for a long time? coresponding to loss of memory
# 6) consider a Geometric distribution as prior (i.e. in the model) over number of options (K) - many times there only exist 1 option, but sometimes we have an unknown number of options (at least at compile time)
# 7) consider a Geometric distribution as prior (i.e. in the model) over number of planning timesteps (T) - often it is sufficient to only plan a few time steps into the future (fast inference), e.g. when exploring, however, sometimes it would be advantageous to plan further into the future (high "T" mean slower inference), e.g. when planning to reach a specific state.
# 8) rejection sampling f