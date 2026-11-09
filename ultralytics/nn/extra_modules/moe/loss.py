import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELoss(nn.Module):
    



    def __init__(self, balance_loss_coeff=0.01, z_loss_coeff=1e-3, num_experts=4, top_k=2):
        super().__init__()
        self.balance_loss_coeff = balance_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, router_probs, router_logits, expert_indices):
        








        
        
        importance = router_probs.mean(dim=0) 
        
        
        usage_mask = torch.zeros_like(router_probs)
        
        
        for k in range(self.top_k):
            usage_mask.scatter_(1, expert_indices[:, k].unsqueeze(1), 1.0)
        usage = usage_mask.mean(dim=0)
        
        balance_loss = self.num_experts * torch.sum(importance * usage.detach())
        
        
        
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=1)**2)
        
        return (self.balance_loss_coeff * balance_loss) + (self.z_loss_coeff * z_loss)
    
        
