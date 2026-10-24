import math
import torch
from torch import nn
from torch.nn import functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=(128,128)):
        super().__init__()
        self.obs_dim, self.hidden = obs_dim, list(hidden)
        def mlp(out):
            layers=[]; prev=obs_dim
            for width in hidden:
                layers += [nn.Linear(prev,width), nn.Tanh()]; prev=width
            layers += [nn.Linear(prev,out)]
            return nn.Sequential(*layers)
        self.actor, self.critic = mlp(5), mlp(1)
        self.log_std = nn.Parameter(torch.full((5,), -1.0))
        self.register_buffer('mid', torch.tensor([0.,0.,.5,.5,.5]))
        self.register_buffer('scale', torch.tensor([1.,1.,.5,.5,.5]))
        # Native sensor scales are the immutable observation transform. No running normalization.
        self.register_buffer('obs_offset', torch.zeros(obs_dim))
        self.register_buffer('obs_scale', torch.ones(obs_dim))
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2));nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01)
        with torch.no_grad(): self.actor[-1].bias.copy_(torch.tensor([0.,.55,-3.,-3.,-3.]))
        nn.init.orthogonal_(self.critic[-1].weight,1.)
    def transform(self, obs):
        return (obs-self.obs_offset)*self.obs_scale
    def distribution(self, obs):
        return torch.distributions.Normal(self.actor(self.transform(obs)), self.log_std.clamp(-5,2).exp())
    def value(self,obs):
        return self.critic(self.transform(obs)).squeeze(-1)
    def action(self,obs,deterministic=False):
        dist=self.distribution(obs)
        u=dist.mean if deterministic else dist.sample()
        return self.mid+self.scale*u.tanh(),u,self.log_prob(obs,u),self.value(obs)
    def log_prob(self,obs,u):
        # log(1-tanh(u)^2) = 2*(log(2)-u-softplus(-2u)), stable at saturation.
        jac=2*(math.log(2)-u-F.softplus(-2*u))
        return (self.distribution(obs).log_prob(u)-self.scale.log()-jac).sum(-1)
    def reject_legacy_export(self):
        raise ValueError('native_rl_v1 includes extra observations and affine unsigned controls; legacy JSON cannot represent it')
