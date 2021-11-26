from abc import ABC, abstractmethod
from misc import KL_point_estimate, Lautum_information_estimate, probabilistic_OR_independent, probabilistic_AND_independent, gradient_modifier

import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoname import scope

class Planning(ABC):

    params = {}
    LTM = {}
    p_z_s_Minus = []

    def __init__(self,
                 K,  # K: number of options/trajectories to consider
                 M,  # M: number of samples from each independent perception in calculation of the information gain
                 N,  # N: number of LTM samples used in calculation of the information gain
                 G,  # G: number of samples from each independent constraint
                 svi_epochs,
                 optimizer,
                 desirability_scale_factor=1,
                 progress_scale_factor=1,
                 info_gain_scale_factor=1,
                 loss=None):

        self.K = K
        self.M = M
        self.N = N
        self.G = G

        self.svi_epochs = svi_epochs

        self.params["desirability_scale_factor"] = torch.tensor([desirability_scale_factor], dtype=torch.float)
        self.params["progress_scale_factor"] = torch.tensor([progress_scale_factor], dtype=torch.float)
        self.params["info_gain_scale_factor"] = torch.tensor([info_gain_scale_factor], dtype=torch.float)
        self.params["P_z_p_max"] = 1.0
        self.params["P_z_p_decay_max"] = 0.5
        self.params["N_old_states_to_consider"] = 5
        self.params["P_z_i_max"] = 1.0

        # In case of vanishing gradients try to modify this
        self.information_gradient_multiplier = torch.tensor(1.)
        self.constraint_gradient_multiplier = torch.tensor(1.)
        self.p_z_g = None

        # https://pyro.ai/examples/svi_part_iv.html
        if loss is None:
            loss = pyro.infer.TraceEnum_ELBO(num_particles=1)

        self.svi_instance = pyro.infer.SVI(model=self.__WM_planning_model,
                                           guide=self.__WM_planning_guide,
                                           optim=optimizer,
                                           loss=loss)

    def makePlan(self, t, T_delta, p_z_s_t, LTM, N_posterior_samples=1, p_z_g=None):
        # T_delta: number of timesteps to predict into to future
        self.T_delta = T_delta
        T = torch.tensor([t + self.T_delta])

        # update the LTM store
        self.LTM = LTM

        self.p_z_g = p_z_g

        # add current state distribution to p_z_s_Minus and maybe delete TOO old state distributions that will not be used anymore!
        # self.p_z_s_Minus.append(p_z_s_t)
        self.p_z_s_Minus.append(poutine.trace(p_z_s_t).get_trace())
        # self.p_z_s_Minus.append(poutine.trace(p_z_s_t).get_trace().nodes["z_s"]["fn"])
        self.params["N_old_states"] = len(self.p_z_s_Minus)
        if len(self.p_z_s_Minus) > self.params["N_old_states_to_consider"]:
            del self.p_z_s_Minus[0]

        # when introducing goal consider to scale the scale factors for progress and information gain
        # relative to the initial desirability D_KL, i.e. scale with a baseline:
        # self.params["baseline"] = KL_point_estimate(z_s_0_mean, p_z_s_0, P_g)

        # take svi steps...
        pyro.clear_param_store()
        losses = []
        for svi_epoch in range(self.svi_epochs):
            step_loss = self.svi_instance.step(t, T, p_z_s_t, self.p_z_s_Minus)
            losses.append(step_loss)
            # print("svi_epoch: " + str(svi_epoch) + "    loss: " + str(step_loss), flush=True)

        # sample the next action according to the posterior guide
        z_a_tauPlus_samples = []
        z_s_tauPlus_samples = []
        for i in range(N_posterior_samples):
            z_a_tauPlus, z_s_tauPlus, k = self.__WM_planning_guide(t, T, p_z_s_t, self.p_z_s_Minus)
            z_a_tauPlus_samples.append(z_a_tauPlus)
            z_s_tauPlus_samples.append(z_s_tauPlus)

        return z_a_tauPlus_samples, z_s_tauPlus_samples

    def reset(self):
        self.LTM = {}
        self.p_z_s_Minus = []
        pyro.clear_param_store()

    def __WM_planning_model(self, t, T, p_z_s_t, p_z_s_Minus):
        _p_z_s_Minus = p_z_s_Minus.copy()

        with scope(prefix=str(t)):
            p_z_s_t_trace = poutine.trace(p_z_s_t).get_trace()
            z_s_t = p_z_s_t_trace.nodes["_RETURN"]["value"]

        P_z_C_accum = torch.tensor([1.], dtype=torch.float)

        assignment_probs = torch.ones(self.K) / self.K
        k = pyro.sample('k', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})
        # k is only used in the guide, but due to Pyro it also needs to be in the model

        # P_impasse = torch.tensor([1.0]) - self.__P_z_p_tau(t, p_z_s_t_trace, _p_z_s_Minus[0:-2], decay=1.0)
        P_impasse = torch.tensor([1.0]) - self.__P_z_p_tau(t, p_z_s_t_trace, _p_z_s_Minus[0], decay=1.0)
        # P_impasse = torch.tensor([0.1])

        # sample planning steps recursively
        z_a_tauPlus, z_s_tauPlus, P_z_d_end = self.__WM_planning_step_model(t + 1, T, k, z_s_t, _p_z_s_Minus, P_z_C_accum, P_impasse)
        z_s_tauPlus.insert(0, z_s_t)
        return z_a_tauPlus, z_s_tauPlus, k

    def __WM_planning_guide(self, t, T, p_z_s_t, p_z_s_Minus):
        with scope(prefix=str(t)):
            p_z_s_t_trace = poutine.trace(p_z_s_t).get_trace()
            z_s_t = p_z_s_t_trace.nodes["_RETURN"]["value"]

        assignment_probs = pyro.param('assignment_probs', torch.ones(self.K) / self.K, constraint=constraints.unit_interval)
        k = pyro.sample('k', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})
        z_a_tauPlus, z_s_tauPlus = self.__WM_planning_step_guide(t + 1, T, k, z_s_t)
        z_s_tauPlus.insert(0, z_s_t)
        return z_a_tauPlus, z_s_tauPlus, k

    def __WM_planning_step_model(self, tau, T, k, z_s_tauMinus1, p_z_s_Minus, P_z_C_accum, P_impasse):
        with scope(prefix=str(tau)):
            z_a_tauMinus1 = self.p_z_a_tau(z_s_tauMinus1)

            p_z_s_tau_trace = poutine.trace(self.p_z_s_tau).get_trace(z_s_tauMinus1, z_a_tauMinus1)
            z_s_tau = p_z_s_tau_trace.nodes["_RETURN"]["value"]

        # calculate the (pseudo) probability of the state giving new information
        # P_z_i = self.__P_z_i_tau(z_s_tau)

        # calculate the (pseudo) probability of the state yielding progress compared to previous states
        # P_z_p = self.__P_z_p_tau(tau, p_z_s_tau_trace, p_z_s_Minus)

        # calculate the (pseudo) probability of the state violating constraints and accummulate that probability
        P_z_C_tau = self.__P_z_c_tau(z_s_tau, z_s_tauMinus1)
        P_z_C_accum = probabilistic_AND_independent([P_z_C_accum, P_z_C_tau])

        if tau >= T:
            P_z_d = self.__P_z_d_tau(tau, p_z_s_tau_trace, self.p_z_g)

            # print("P_z_d: " + str(P_z_d) + "    P_impasse: " + str(P_impasse))

            # if P_z_d < P_impasse:  # if pseudo probability of being close to the goal is smaller than the probability of being in an impasse, then explore
            if P_impasse > torch.tensor([0.90]):
                P_z_d = torch.tensor([0.0])

            if P_z_d < torch.tensor([0.10]):  # if pseudo probability of being close to the goal is small, then explore
                # calculate the (pseudo) probability of the state giving new information
                P_z_i = self.__P_z_i_tau(z_s_tau)

                # calculate the (pseudo) probability of the state yielding progress compared to previous states
                P_z_p = self.__P_z_p_tau(tau, p_z_s_tau_trace, p_z_s_Minus)
            else:
                P_z_i = torch.tensor([0.0])
                P_z_p = torch.tensor([0.0])

            with scope(prefix=str(tau)):
                pyro.sample("x_A", self.__p_z_A_tau(P_z_d, P_z_p, P_z_i, P_z_C_accum), obs=torch.tensor([1.], dtype=torch.float))

            z_a_tauPlus = [z_a_tauMinus1]
            z_s_tauPlus = [z_s_tau]
            return z_a_tauPlus, z_s_tauPlus, P_z_d
        else:
            z_a_tauPlus, z_s_tauPlus, P_z_d_end = self.__WM_planning_step_model(tau + 1, T, k, z_s_tau, p_z_s_Minus, P_z_C_accum, P_impasse)
            # consider constraining this on if the end state of this trajectory actually ended up finding at the goal...
            # i.e. if it did, we e.g. might not care about progress (_p_z_P)

            # if P_z_d_end < P_impasse:  # if pseudo probability of being close to the goal is smaller than the probability of being in an impasse, then explore
            if P_z_d_end < torch.tensor([0.10]):  # if pseudo probability of being close to the goal is small, then explore
                # calculate the (pseudo) probability of the state giving new information
                P_z_i = self.__P_z_i_tau(z_s_tau)

                # calculate the (pseudo) probability of the state yielding progress compared to previous states
                P_z_p = self.__P_z_p_tau(tau, p_z_s_tau_trace, p_z_s_Minus)
            else:
                P_z_i = torch.tensor([0.0])
                P_z_p = torch.tensor([0.0])

            with scope(prefix=str(tau)):
                pyro.sample("x_A", self.__p_z_A_tau(P_z_d_end, P_z_p, P_z_i, P_z_C_accum), obs=torch.tensor([1.], dtype=torch.float))

            z_a_tauPlus.insert(0, z_a_tauMinus1)
            z_s_tauPlus.insert(0, z_s_tau)
            return z_a_tauPlus, z_s_tauPlus, P_z_d_end

    def __WM_planning_step_guide(self, tau, T, k, z_s_tauMinus1):
        with scope(prefix=str(tau)):
            z_a_tauMinus1 = self.q_z_a_tau(z_s_tauMinus1, k)

            p_z_s_tau_trace = poutine.trace(self.p_z_s_tau).get_trace(z_s_tauMinus1, z_a_tauMinus1)
            z_s_tau = p_z_s_tau_trace.nodes["_RETURN"]["value"]

        if tau >= T:
            z_a_tauPlus = [z_a_tauMinus1]
            z_s_tauPlus = [z_s_tau]
            return z_a_tauPlus, z_s_tauPlus
        else:
            z_a_tauPlus, z_s_tauPlus = self.__WM_planning_step_guide(tau + 1, T, k, z_s_tau)
            z_a_tauPlus.insert(0, z_a_tauMinus1)
            z_s_tauPlus.insert(0, z_s_tau)
            return z_a_tauPlus, z_s_tauPlus

    def __P_z_A_tau(self, P_z_d, P_z_p, P_z_i, P_z_c):
        P_z_A1 = probabilistic_OR_independent([P_z_i, P_z_p, P_z_d])  # <-- the order of args might matter!
        P_z_A = probabilistic_AND_independent([P_z_A1, P_z_c])

        # P_z_A = probabilistic_AND_independent([P_z_p, P_z_c])
        # P_z_A = P_z_p
        # P_z_A = probabilistic_AND_independent([P_z_i, P_z_c])
        # P_z_A = P_z_i
        return P_z_A

    def __p_z_A_tau(self, P_z_d, P_z_p, P_z_i, P_z_c):
        P_z_A = self.__P_z_A_tau(P_z_d, P_z_p, P_z_i, P_z_c)
        # print("P_z_d: " + "{:.6f}".format(P_z_d.item()) + "  P_z_p: " + "{:.6f}".format(P_z_p.item()) +
        #       "  P_z_i: " + "{:.6f}".format(P_z_i.item()) + "  P_z_c: " + "{:.6f}".format(P_z_c.item()) +
        #       "  P_z_A: " + "{:.6f}".format(P_z_A.item()))
        return dist.Bernoulli(P_z_A)

    def __P_z_d_tau(self, tau, p_z_s_tau_trace, p_z_g):
        if p_z_g is not None:
            # calculate kl-divergence
            with poutine.block():
                p_z_g_trace = poutine.trace(p_z_g).get_trace()
            KL_estimate = KL_point_estimate(tau, p_z_s_tau_trace, p_z_g_trace)

            # calculate "pseudo" probability
            P_z_d = torch.exp(-self.params["desirability_scale_factor"] * KL_estimate)
        else:
            P_z_d = torch.tensor(0.0)

        return P_z_d

    def __P_z_p_tau(self, tau, p_z_s_tau_trace, p_z_s_Minus, decay=None):
        # make some optimization + add the different parameters to the param dict!
        # if hasattr(p_z_s_Minus, '__iter__'):
        if not isinstance(p_z_s_Minus, pyro.poutine.trace_struct.Trace):
            if self.params["N_old_states_to_consider"] > len(p_z_s_Minus):
                N_old_states_to_consider = len(p_z_s_Minus)
            else:
                N_old_states_to_consider = self.params["N_old_states_to_consider"]
            P_z_p_list = []
            for i in range(N_old_states_to_consider):  # optimize!!!
                idx = len(p_z_s_Minus) - i
                KL_estimate = KL_point_estimate(tau, p_z_s_tau_trace, p_z_s_Minus[idx - 1])

                if True:  # decay is None:
                    decay = 1 - (1 - self.params["P_z_p_decay_max"]) * (N_old_states_to_consider - i) / N_old_states_to_consider
                # print("len(p_z_s_Minus): " + str(len(p_z_s_Minus)) + "  i: " + str(i) + "   idx: " + str(idx) + "    decay: " + str(decay))
                P_z_p_list.append(decay * torch.exp(-self.params["progress_scale_factor"] * KL_estimate))
            P_z_p = torch.tensor([1.], dtype=torch.float) - probabilistic_OR_independent(P_z_p_list)
        else:
            KL_estimate = KL_point_estimate(tau, p_z_s_tau_trace, p_z_s_Minus)
            P_z_p = torch.tensor([1.], dtype=torch.float) - torch.exp(-self.params["progress_scale_factor"] * KL_estimate)

        P_z_p = P_z_p * self.params["P_z_p_max"]
        return P_z_p

    def __P_z_i_tau(self, z_s_tau):
        with poutine.block():
            _z_s_tau = z_s_tau.detach()
            _z_s_tau.requires_grad = True

            # condition all the relevant distributions on the current state sample, z_s_tau:
            def p_z_1_prior():
                return self.p_z_LTM(_z_s_tau)

            def p_z_2_prior():
                return self.p_z_PB(_z_s_tau)

            def p_z_2_posterior(z_LTM):
                return self.p_z_PB_posterior(_z_s_tau, z_LTM)

            # Fetch labels/keys to use as observation sites
            observation_labels = self.generate_obs_labels()  # might contain pyro.sample statements!

            # Calculate the information gain
            information = Lautum_information_estimate(p_z_1_prior, p_z_2_prior, p_z_2_posterior, observation_labels, M=self.M, N=self.N)

            # if len(observation_labels)>0 it might be possible that it is possible to obtain information in different directions, i.e.
            # the gradients for each of these observations might be working against each other. Therefore, we only seek in the direction
            # with most information:
            information_max = torch.max(information)

            _P_z_i = torch.tensor([1.], dtype=torch.float) - torch.exp(-self.params["info_gain_scale_factor"] * information_max)
            _P_z_i = _P_z_i * self.params["P_z_i_max"]

            if _P_z_i.requires_grad:
                _P_z_i.backward()
                z_s_tau_grad = _z_s_tau.grad
            else:
                z_s_tau_grad = None
            P_z_i = _P_z_i.detach()

        P_z_i_out = gradient_modifier.apply(z_s_tau, P_z_i, z_s_tau_grad, self.information_gradient_multiplier)

        return P_z_i_out

    def __P_z_c_tau(self, z_s_tau, z_s_tauMinus1):
        with poutine.block():
            _z_s_tau = z_s_tau.detach()
            _z_s_tau.requires_grad = True
            for g in range(self.G):
                I_c_tau = self.I_c_tau(_z_s_tau, z_s_tauMinus1.detach())
                if g == 0:
                    _P_z_c_tau_ = torch.zeros(len(I_c_tau))
                for i in range(len(I_c_tau)):
                    _P_z_c_tau_[i] = _P_z_c_tau_[i] + I_c_tau[i]

            for i in range(len(_P_z_c_tau_)):
                _P_z_c_tau_[i] = _P_z_c_tau_[i] / self.G

            _P_z_c_tau = probabilistic_AND_independent(_P_z_c_tau_)

            if _P_z_c_tau.requires_grad:
                _P_z_c_tau.backward()
                z_s_tau_grad = _z_s_tau.grad
            else:
                z_s_tau_grad = None
            P_z_c_tau = _P_z_c_tau.detach()

        P_z_c_tau = gradient_modifier.apply(z_s_tau, P_z_c_tau, z_s_tau_grad, self.constraint_gradient_multiplier)

        return P_z_c_tau

    # ############### Methods that needs to be implemented by the user! ###############
    # decide if the following methods should instead be classes inhereting from pytorch or pyro distributions!
    @abstractmethod
    def q_z_a_tau(self, z_s_tauMinus1, k):
        # should return a probability distribution class with the following methods
        #   .sample()
        #   .log_prob(z)
        raise NotImplementedError

    @abstractmethod
    def p_z_a_tau(self, z_s_tau):
        # should return a probability distribution class with the following methods
        #   .sample()
        #   .log_prob(z)
        raise NotImplementedError

    @abstractmethod
    def p_z_s_tau(self, z_s_tauMinus1, z_a_tauMinus1):
        # should return a probability distribution class with the following methods
        #   .sample()
        #   .log_prob(z)
        raise NotImplementedError

    # @abstractmethod
    # def P_z_c_tau(self, z_s_tau, z_s_tauMinus1):
    #     # returns list of constraint probabilities on the form
    #     # 1 - e^(-k * c_i(z_s_tau, z_s_tauMinus1)) for maximization of c_i(...)
    #     # or
    #     # e^(-k * c_i(z_s_tau, z_s_tauMinus1)) for minimization of c_i(...)
    #     # where c_i(...) are non-negative functions
    #     raise NotImplementedError

    @abstractmethod
    def I_c_tau(self, z_s_tau, z_s_tauMinus1):
        # returns list of outputs of constraint indicator functions taking the args:
        # z_s_tau, z_s_tauMinus1
        # That is the function should return a list like:
        # [I_c1(z_s_tau, z_s_tauMinus1), ... , I_c10(z_s_tau, z_s_tauMinus1)]
        raise NotImplementedError

    @abstractmethod
    def p_z_LTM(self, z_s_tau):
        # z_s_tau should not be used to alter the distribution over LTM!
        # z_s_tau is only included for the purpose of optimizing which part of LTM
        # that is being sampled
        raise NotImplementedError

    @abstractmethod
    def p_z_PB(self, z_s_tau):
        raise NotImplementedError

    @abstractmethod
    def p_z_PB_posterior(self, z_s_tau, z_LTM):
        raise NotImplementedError

    @abstractmethod
    def generate_obs_labels(self):
        # function that returns a list of labels/keys specifying the sample sites in
        # p_z_PB(...) that should be considered observations for in the calculation of
        # information gain in __P_z_i_tau(...)
        raise NotImplementedError


if __name__ == '__main__':
    a = 1