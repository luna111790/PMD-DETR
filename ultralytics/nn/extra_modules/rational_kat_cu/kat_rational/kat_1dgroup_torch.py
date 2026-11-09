import torch
from torch import nn
import os
import json


def _get_xps(z, len_numerator, len_denominator):
    











    xps = [z]
    for _ in range(max(len_numerator, len_denominator) - 2):
        xps.append(xps[-1] * z)
    xps.insert(0, torch.ones_like(z))  
    return torch.stack(xps, dim=1)


def Rational_CUDA_A_1DGroup(x, weight_numerator, weight_denominator, group):
    















    device = x.device
    B, L, D = x.shape
    len_num = weight_numerator.size(1)
    len_deno = weight_denominator.size(1)

    
    D_per_group = D // group

    
    z = x.view(B, L, group, D_per_group).permute(2, 0, 1, 3).contiguous()  
    z = z.view(group, B * L * D_per_group)  

    
    xps = _get_xps(z, len_num, len_deno)  

    
    numerator = torch.bmm(weight_numerator.unsqueeze(1), xps).squeeze(1)  

    
    expanded_dw = torch.cat([
        torch.ones(group, 1, device=device),  
        weight_denominator,
        torch.zeros(group, max(0, len_num - len_deno - 1), device=device)  
    ], dim=1)

    denominator = torch.bmm(expanded_dw.abs().unsqueeze(1), xps).squeeze(1)  

    
    result = numerator.div(denominator)

    
    result = result.view(group, B, L, D_per_group).permute(1, 2, 0, 3).contiguous()  
    result = result.view(B, L, D)  

    return result


class KAT_Group_Torch(nn.Module):
    def __init__(self, num_groups=8, mode="gelu"):
        






        super(KAT_Group_Torch, self).__init__()
        self.order = (5, 4)
        self.num_groups = num_groups
        
        self.initialize(mode=mode)
        
    def init_info(self):
        

        cfd = os.path.dirname(os.path.realpath(__file__))
        with open(f'{cfd}/init.json') as json_file:
            data = json.load(json_file)
        return data
                
    def initialize(self, mode="gelu"):
        





        cfd = os.path.dirname(os.path.realpath(__file__))
        try:
            with open(f'{cfd}/init.json') as json_file:
                data = json.load(json_file)
            weight_numerator = torch.tensor(data[mode]["init_w_numerator"])
            weight_numerator = torch.cat([weight_numerator]).view(1, -1)
            weight_denominator = torch.tensor(data[mode]["init_w_denominator"])
            weight_denominator = torch.cat([weight_denominator]*self.num_groups).view(self.num_groups, -1)
             
            self.weight_numerator = nn.Parameter(torch.FloatTensor(weight_numerator)
                                                      , requires_grad=True) 
            self.weight_denominator = nn.Parameter(torch.FloatTensor(weight_denominator)
                                                      , requires_grad=True) 

        except FileNotFoundError:
            print("Initialization JSON file not found.")
        except json.JSONDecodeError:
            print("Error decoding JSON.")
            
    def forward(self, input):
        









        assert input.dim() == 3, "Input tensor must be 3D. Of size (batch, length, channels)."
    
        
        weight_numerator = self.weight_numerator.repeat(self.num_groups, 1)
        return Rational_CUDA_A_1DGroup(input, weight_numerator, self.weight_denominator, self.num_groups)
        
    
    def extra_repr(self):
        





        return f'num_groups={self.num_groups}, order={self.order}'
    