import torch
import pyro
from pyro import poutine
from pyro.contrib.autoname import scope
relu = torch.nn.ReLU()


def KL_point_estimate(tau, p1_trace, p2_trace, D_KL_baseline=torch.tensor([1.], dtype=torch.float)):
    p1_trace_log_prob_sum = torch.tensor(0.0)
    p2_trace_log_prob_sum = torch.tensor(0.0)
    for p1_key in p1_trace:
        # p2_key = p1_key.removeprefix(str(tau) + "/")
        p2_key = p1_key.replace(str(tau) + "/", "")
        if p2_trace.nodes[p2_key]["type"] == "sample":
            p1_trace_log_prob_sum = p1_trace_log_prob_sum + p1_trace.nodes[p1_key]["fn"].log_prob(p1_trace.nodes[p1_key]["value"]).sum()
            p2_trace_log_prob_sum = p2_trace_log_prob_sum + p2_trace.nodes[p2_key]["fn"].log_prob(p1_trace.nodes[p1_key]["value"]).sum()

    KL_point_est = p1_trace_log_prob_sum - p2_trace_log_prob_sum
    KL_point_est = KL_point_est / D_KL_baseline.detach()
    KL_point_est = relu(KL_point_est)
    return KL_point_est


def Lautum_information_estimate(p_z_1_prior, p_z_2_prior, p_z_2_posterior, observation_labels, M=1, N=3):
    N_obs = len(observation_labels)
    information = torch.zeros(N_obs)
    outerscope = ""
    scale = float(M * N * N_obs)

    z_1_ = []
    for n in range(N):
        with poutine.block():
            with scope(prefix=str(n)):
                z_1 = p_z_1_prior()
                z_1_.append(z_1)

    for m in range(M):
        p_z_2_posterior_log_probs = torch.zeros(N_obs, N)

        with scope(prefix=str(m)):
            with poutine.block():
                p_z_2_prior_trace = poutine.trace(p_z_2_prior).get_trace()

            for n in range(N):
                with scope(prefix=str(n)):
                    obs_dict = {key: p_z_2_prior_trace.nodes[key]["value"] for key in observation_labels}

                    for obs in obs_dict:
                        obs_dict[obs].requires_grad = True
                    conditional_model = pyro.condition(p_z_2_posterior, data=obs_dict)
                    conditional_model = poutine.scale(conditional_model, 1 / scale)  # scale all sample sites to get reasonable log_probs in nested inference
                    p_z_2_posterior_trace = poutine.trace(conditional_model).get_trace(z_1_[n])
                    p_z_2_posterior_trace.compute_log_prob()

                    if m == 0 and n == 0:  # determine name for possible outer scopes!
                        posterior_keys = list(name for name in p_z_2_posterior_trace.nodes.keys() if name.endswith(observation_labels[0]))
                        outerscope = posterior_keys[0].replace(str(m) + "/" + str(n) + "/" + observation_labels[0], "")

                    for i in range(N_obs):
                        key = observation_labels[i]
                        p_z_2_posterior_log_probs[i, n] = p_z_2_posterior_trace.nodes[outerscope + str(m) + "/" + str(n) + "/" + key]["log_prob_sum"] * scale  # scale log_probs back to get prober info estimate

        for i in range(N_obs):
            information[i] = information[i] + torch.logsumexp(p_z_2_posterior_log_probs[i], 0) - torch.log(torch.tensor(N)) - torch.sum(p_z_2_posterior_log_probs[i].detach()) / N  # <-- this ".detach()" improves gradients?

    information = information / M
    information = relu(information)
    return information


def probabilistic_OR_independent(p_list):
    p_OR = torch.tensor([0.], dtype=torch.float)
    for p in p_list:
        p_OR = p_OR + p - (p_OR * p)
    return p_OR


def probabilistic_AND_independent(p_list):
    p_AND = torch.tensor([1.], dtype=torch.float)
    for p in p_list:
        p_AND = p_AND * p
    return p_AND


class gradient_modifier(torch.autograd.Function):
    # consider making more general!
    # currently probably only works for vector --> scalar!
    @staticmethod
    def forward(ctx, _input, _output, gradient, gradient_multiplier=torch.tensor(1.0)):
        ctx.save_for_backward(gradient, _input, _output, gradient_multiplier)
        return _output

    @staticmethod
    def backward(ctx, grad_output):
        gradient, _input, _output, gradient_multiplier = ctx.saved_tensors
        if grad_output is not None and gradient is not None:
            grad_input = gradient_multiplier * gradient * grad_output
        else:
            grad_input = None

        return grad_input, None, None, None