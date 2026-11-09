import torch
import torch.nn as nn
from .routers import *
from .experts import *
from .loss import MoELoss

class ESMoE(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_type: str = 'simple',   
        router_type: str = 'efficient',
        noise_std: float = 1.0,
        balance_loss_coeff: float = 0.01,
        router_z_loss_coeff: float = 1e-3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.router_z_loss_coeff = router_z_loss_coeff
        
        
        if router_type == 'local':
            self.routing = LocalRoutingLayer(in_channels, num_experts, top_k=top_k, noise_std=noise_std)
        elif router_type == 'adaptive':
            self.routing = AdaptiveRoutingLayer(in_channels, num_experts, top_k=top_k, noise_std=noise_std)
        else:
            self.routing = EfficientSpatialRouter(in_channels, num_experts, top_k=top_k, noise_std=noise_std)

        
        self.experts = nn.ModuleList()
        if expert_type == 'ghost':
            expert_cls = GhostExpert
        elif expert_type == 'inverted':
            expert_cls = InvertedResidualExpert
        else:
            expert_cls = SimpleExpert
            
        for _ in range(num_experts):
            self.experts.append(expert_cls(in_channels, out_channels))
            
        
        self.shared_expert = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        self._init_weights()
        self.aux_loss = torch.tensor(0.0)
        self.moe_loss_fn = MoELoss(balance_loss_coeff, router_z_loss_coeff, num_experts, top_k)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        
        
        for m in self.routing.router.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv:
            nn.init.normal_(last_conv.weight, mean=0, std=0.01)
            if last_conv.bias is not None:
                nn.init.constant_(last_conv.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        
        
        
        routing_weights, routing_indices, loss_dict = self.routing(x)
        
        
        shared_out = self.shared_expert(x)
        
        
        
        expert_output = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)
        
        indices_flat = routing_indices.view(B, self.top_k)
        weights_flat = routing_weights.view(B, self.top_k)
        
        for i in range(self.num_experts):
            
            mask = (indices_flat == i)
            if mask.any():
                batch_idx, k_idx = torch.where(mask)
                
                
                inp = x[batch_idx]
                out = self.experts[i](inp)
                
                
                w = weights_flat[batch_idx, k_idx].view(-1, 1, 1, 1)
                
                
                expert_output.index_add_(0, batch_idx, out.to(expert_output.dtype) * w.to(expert_output.dtype))
        
        final_output = shared_out + expert_output

        
        if self.training and loss_dict:
             self.aux_loss = self.moe_loss_fn(loss_dict['router_probs'], loss_dict['router_logits'], loss_dict['topk_indices'])
        else:
             pass

        return final_output