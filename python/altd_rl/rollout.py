import torch

def gae(rewards, values, next_values, terminated, truncated, gamma=1., lam=.95):
    """Bootstrap truncations from their final state; stop recurrence across every reset."""
    adv=torch.zeros_like(rewards); carry=torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*(~terminated[t]).to(values.dtype)*next_values[t]-values[t]
        carry=delta+gamma*lam*(~(terminated[t]|truncated[t])).to(values.dtype)*carry
        adv[t]=carry
    return adv,adv+values

class Collector:
    def __init__(self,env,device='cpu'):
        self.env,self.device=env,device;self.obs=env.observe();self.starts=torch.ones(env.n,dtype=torch.bool)
    @torch.no_grad()
    def collect(self,policy,steps,summary_sink=None,trace_sink=None,policy_version=0):
        rows={k:[] for k in ('obs','normalized_obs','u','actions','old_log_prob','values','next_values','rewards','terminated','truncated','episode_start','valid_transition_mask')}
        ticks=0
        for _ in range(steps):
            obs=torch.as_tensor(self.obs,device=self.device)
            a,u,lp,v=policy.action(obs)
            b=self.env.step(a.cpu().numpy()); nxt=torch.as_tensor(b['next_observations'],device=self.device)
            term=torch.as_tensor(b['terminated'],device=self.device,dtype=torch.bool)
            trunc=torch.as_tensor(b['truncated'],device=self.device,dtype=torch.bool)
            nv=policy.value(nxt); nv[term]=0
            values=dict(obs=obs,normalized_obs=policy.transform(obs),u=u,actions=a,old_log_prob=lp,values=v,next_values=nv,
                        rewards=torch.as_tensor(b['rewards'],device=self.device,dtype=torch.float32),terminated=term,truncated=trunc,
                        episode_start=self.starts.to(self.device),valid_transition_mask=torch.ones_like(term))
            for k,x in values.items():rows[k].append(x.clone())
            ticks+=sum(i['tick_end']-i['tick_start'] for i in b['info'])
            if trace_sink:trace_sink.append(b, values, policy_version)
            for i,ended in enumerate(b['terminated']|b['truncated']):
                if ended and summary_sink:summary_sink(b['info'][i])
            self.obs=b['next_observations'].copy()
            ids=(b['terminated']|b['truncated']).nonzero()[0]
            if len(ids):self.obs[ids]=self.env.reset(ids)
            self.starts=(term|trunc).cpu()
        return {k:torch.stack(v) for k,v in rows.items()},ticks
