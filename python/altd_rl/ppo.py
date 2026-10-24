import torch
from .rollout import gae

def update(policy,optimizer,b,cfg,gamma=1.,epoch_callback=None):
    adv,ret=gae(b['rewards'],b['values'],b['next_values'],b['terminated'],b['truncated'],gamma,cfg['gae_lambda'])
    flat={k:v.flatten(0,1) for k,v in b.items()}
    adv,ret=adv.flatten(),ret.flatten()
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    n=len(adv);metrics=[];stop=False
    for epoch in range(cfg['epochs']):
        for ids in torch.randperm(n,device=adv.device).split(cfg['minibatch_size']):
            new=policy.log_prob(flat['obs'][ids],flat['u'][ids]); logratio=new-flat['old_log_prob'][ids]
            ratio=logratio.exp();kl=((ratio-1)-logratio).mean()
            if not torch.isfinite(kl):raise FloatingPointError('nonfinite PPO KL')
            if kl.item()>cfg['target_kl']:stop=True;break
            actor=-torch.min(ratio*adv[ids],ratio.clamp(1-cfg['clip_epsilon'],1+cfg['clip_epsilon'])*adv[ids]).mean()
            value=policy.value(flat['obs'][ids]);vf=.5*(value-ret[ids]).square().mean()
            # Monte Carlo transformed-policy entropy; no latent Gaussian entropy substitution.
            sample=policy.distribution(flat['obs'][ids]).rsample();entropy=-policy.log_prob(flat['obs'][ids],sample).mean()
            loss=actor+cfg['value_coefficient']*vf-cfg['entropy_coefficient']*entropy
            optimizer.zero_grad();loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),cfg['max_grad_norm'])
            if not torch.isfinite(grad):raise FloatingPointError('nonfinite PPO gradient')
            optimizer.step()
            metrics.append([actor.item(),vf.item(),kl.item(),entropy.item(),((ratio-1).abs()>cfg['clip_epsilon']).float().mean().item()])
        if epoch_callback is not None:epoch_callback(epoch+1)
        if stop:break
    result=dict(zip(('actor_loss','value_loss','kl','entropy','clip_fraction'),torch.tensor(metrics).mean(0).tolist())) if metrics else {}
    result['epochs_completed']=epoch+1
    result['kl_early_stop']=stop
    result['explained_variance']=float(1-(ret-flat['values']).var(unbiased=False)/(ret.var(unbiased=False)+1e-8))
    return result
