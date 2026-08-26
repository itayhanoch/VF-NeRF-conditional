"""Conditional RealNVP flow engine, ported from itayhanoch/conditional-normalizing-flows-toy
(src/condflow/core/{layers,flow}.py, MIT license), which itself builds on
https://github.com/ikostrikov/pytorch-flows. Kept close to the original so the
architecture matches the toy repo's `dual_cond_rgb` variant (coupling layers AND
the base-distribution prior both conditioned on a continuous context vector).

Ported rather than taken as a package dependency: the toy repo's own pyproject.toml
pins torch>=2.0, while this project's pins torch<2.0 (tiny-cuda-nn/nerfacc
compatibility). Nothing here uses a torch>=2.0-only API, so the source runs fine
under either pin without a package-dependency conflict.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal

EPS = 1e-6


class ConditionProjectionLayer(nn.Module):
    """An MLP layer that maps condition inputs to a target space (e.g., mean or log-var)."""

    def __init__(self, num_outputs, num_cond_inputs, num_hidden, act="tanh", last_act=None, bound=None):
        super().__init__()
        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        act_func = activations[act]

        layers = [
            nn.Linear(num_cond_inputs, num_hidden),
            act_func(),
            nn.Linear(num_hidden, num_hidden),
            act_func(),
            nn.Linear(num_hidden, num_outputs),
        ]

        if last_act == "softplus":
            layers.append(nn.Softplus())
        elif last_act == "tanh":
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self.bound = bound

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        out = self.net(cond_inputs)
        if self.bound is not None:
            out = self.bound * out
        return out, torch.zeros(out.size(0), 1, device=out.device)


class BatchNormFlow(nn.Module):
    def __init__(self, num_inputs, momentum=0.0, eps=1e-5, clamp_gamma=False):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps
        self.log_scale_max = 3
        self.clamp_gamma = clamp_gamma
        self.register_buffer("running_mean", torch.zeros(num_inputs))
        self.register_buffer("running_var", torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        log_gamma = self.log_scale_max * torch.tanh(self.log_gamma / self.log_scale_max) if self.clamp_gamma else self.log_gamma
        if mode == "direct":
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (inputs - self.batch_mean).pow(2).mean(0) + self.eps
                self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
                self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))
                mean, var = self.batch_mean, self.batch_var
            else:
                mean, var = self.running_mean, self.running_var
            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(log_gamma) * x_hat + self.beta
            return y, (log_gamma - 0.5 * torch.log(var + EPS)).sum(-1, keepdim=True)
        else:
            mean, var = (self.batch_mean, self.batch_var) if self.training else (self.running_mean, self.running_var)
            x_hat = (inputs - self.beta) / torch.exp(log_gamma)
            y = x_hat * var.sqrt() + mean
            return y, (-log_gamma + 0.5 * torch.log(var + EPS)).sum(-1, keepdim=True)


class CouplingLayer(nn.Module):
    def __init__(self, num_inputs, num_hidden, mask, num_cond_inputs=None, s_act="tanh", t_act="relu", log_s_max=None):
        super().__init__()
        self.mask = mask
        self.log_s_max = log_s_max
        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        s_act_func, t_act_func = activations[s_act], activations[t_act]
        total_inputs = num_inputs + (num_cond_inputs if num_cond_inputs is not None else 0)
        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden),
            s_act_func(),
            nn.Linear(num_hidden, num_hidden),
            s_act_func(),
            nn.Linear(num_hidden, num_inputs),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden),
            t_act_func(),
            nn.Linear(num_hidden, num_hidden),
            t_act_func(),
            nn.Linear(num_hidden, num_inputs),
        )

    def forward(self, inputs, cond_inputs=None, mode="direct", return_details=False):
        mask = self.mask
        masked_inputs = inputs * mask
        if cond_inputs is not None and self.scale_net[0].in_features > self.mask.shape[0]:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

        if mode == "direct":
            raw_log_s = self.scale_net(masked_inputs)
            log_s = self.log_s_max * torch.tanh(raw_log_s) if self.log_s_max is not None else raw_log_s
            log_s = log_s * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            result, logdet = inputs * torch.exp(log_s) + t, log_s.sum(-1, keepdim=True)
        else:
            raw_log_s = self.scale_net(masked_inputs)
            log_s = self.log_s_max * torch.tanh(raw_log_s) if self.log_s_max is not None else raw_log_s
            log_s = log_s * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            result, logdet = (inputs - t) * torch.exp(-log_s), -log_s.sum(-1, keepdim=True)

        if return_details:
            return result, logdet, {"exp_log_s": torch.exp(log_s), "t": t}
        return result, logdet


class FlowSequential(nn.Sequential):
    def __init__(self, *args, cond_prior=False, mean_net=None, logvar_net=None, num_inputs=2):
        super().__init__(*args)
        # Snapshot the flow-transform modules before mean_net/logvar_net are assigned below,
        # since assigning an nn.Module attribute auto-registers it into self._modules too —
        # forward() must iterate only these, not the prior networks (see get_generator/log_prob,
        # which call mean_net/logvar_net directly; they aren't flow steps).
        self._flow_modules = list(self._modules.values())
        self.cond_prior = cond_prior
        self.mean_net = mean_net
        self.logvar_net = logvar_net
        self.num_inputs = num_inputs

    def forward(self, inputs, cond_inputs=None, mode="direct", logdets=None):
        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)
        modules = self._flow_modules if mode == "direct" else reversed(self._flow_modules)
        for module in modules:
            inputs, logdet = module(inputs, cond_inputs, mode)
            logdets += logdet
        return inputs, logdets

    def log_prob(self, inputs, cond_inputs=None):
        u, log_jacob = self(inputs, cond_inputs)
        if self.cond_prior and self.mean_net is not None and self.logvar_net is not None:
            mean, _ = self.mean_net(None, cond_inputs)
            log_var, _ = self.logvar_net(None, cond_inputs)
            prior = Normal(mean, torch.exp(0.5 * log_var) + EPS)
        else:
            prior = Normal(torch.zeros_like(u), torch.ones_like(u))

        log_probs = prior.log_prob(u)
        return log_probs.sum(-1, keepdim=True) + log_jacob

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        device = next(self.parameters()).device
        if self.cond_prior and self.mean_net is not None and self.logvar_net is not None:
            mean, _ = self.mean_net(None, cond_inputs)
            log_var, _ = self.logvar_net(None, cond_inputs)
            prior = Normal(mean, torch.exp(0.5 * log_var) + EPS)
            if noise is None:
                z = prior.rsample()
            else:
                z = mean + torch.exp(0.5 * log_var) * noise
        else:
            if noise is None:
                z = torch.randn(num_samples, self.num_inputs).to(device)
            else:
                z = noise.to(device)

        return self.forward(z, cond_inputs.to(device) if cond_inputs is not None else None, mode="inverse")[0]


def get_generator(
    num_inputs,
    num_cond_inputs,
    device,
    num_blocks=5,
    num_hidden=1024,
    cond_prior=False,
    use_cond_in_coupling=True,
    use_batchnorm=True,
    log_var_bound=None,
    log_s_max=None,
    clamp_batchnorm_gamma=False,
):
    modules = []
    mean_net = None
    logvar_net = None
    if cond_prior:
        mean_net = ConditionProjectionLayer(num_inputs, num_cond_inputs, num_hidden)
        logvar_net = ConditionProjectionLayer(
            num_inputs,
            num_cond_inputs,
            num_hidden,
            last_act="tanh" if log_var_bound is not None else None,
            bound=log_var_bound,
        )
    coupling_cond_dim = num_cond_inputs if use_cond_in_coupling else None
    mask = (torch.arange(0, num_inputs) % 2).to(device).float()
    for _ in range(num_blocks):
        modules += [CouplingLayer(num_inputs, num_hidden, mask, coupling_cond_dim, log_s_max=log_s_max)]
        if use_batchnorm:
            modules += [BatchNormFlow(num_inputs, clamp_gamma=clamp_batchnorm_gamma)]
        mask = 1 - mask
    return FlowSequential(*modules, cond_prior=cond_prior, mean_net=mean_net, logvar_net=logvar_net, num_inputs=num_inputs).to(device)
