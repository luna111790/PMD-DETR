import torch
import kat_rational_cu
from torch import nn
import os
import json
from .kat_1dgroup_torch import Rational_CUDA_A_1DGroup

class rational_1dgroup(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight_numerator, weight_denominator, group):
        












        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        ctx.group = group
        x = kat_rational_cu.rational_fwd_1dgroup(input, weight_numerator, weight_denominator, group)
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        









        input, weight_numerator, weight_denominator = ctx.saved_tensors
        group = ctx.group
        d_input, d_weight_numerator, d_weight_denominator = kat_rational_cu.rational_bwd_1dgroup(grad_output, input, weight_numerator, weight_denominator, group)
        return d_input, d_weight_numerator, d_weight_denominator, None


class KAT_Group(nn.Module):
    def __init__(self, num_groups=8, mode="gelu", device="cuda"):
        







        super(KAT_Group, self).__init__()
        assert device in ["cuda", "cpu"], "Device must be either 'cuda' or 'cpu'."
        
        self.order = (5, 4)
        self.num_groups = num_groups

        
        self.initialize(mode=mode)
        
        
        if device == "cuda":
            self.rational = rational_1dgroup.apply
        else:
            self.rational = Rational_CUDA_A_1DGroup
            
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

            
            weight_numerator = torch.tensor(data[mode]["init_w_numerator"]).view(1, -1)
            weight_denominator = torch.tensor(data[mode]["init_w_denominator"])
            weight_denominator = torch.cat([weight_denominator] * self.num_groups).view(self.num_groups, -1)
             
            
            self.weight_numerator = nn.Parameter(weight_numerator.float(), requires_grad=True)
            self.weight_denominator = nn.Parameter(weight_denominator.float(), requires_grad=True) 

        except FileNotFoundError:
            print("Initialization JSON file not found.")
        except json.JSONDecodeError:
            print("Error decoding JSON.")
            
    def forward(self, input):
        








        assert input.dim() == 3 or input.dim() == 2, "Input tensor must be 3D (batch, length, channels) or 2D (batch, channels)."
    
        
        weight_numerator = self.weight_numerator.repeat(self.num_groups, 1)
        return self.rational(input, weight_numerator, self.weight_denominator, self.num_groups)
        
    def extra_repr(self):
        





        return f'num_groups={self.num_groups}, order={self.order}'
    
    
if __name__=="__main__":
    
    model = KAT_1DGroup()
    x = torch.linspace(-2, 2, 100)
    
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
    
    x = x.unsqueeze(0).unsqueeze(0)
    y = model(x.cuda())
    x = x.squeeze(0).squeeze(0)
    y = y.squeeze(0).squeeze(0)
    
    import matplotlib.pyplot as plt
    plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy())
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Response of KAT_1DGroup")
    plt.grid(True)
    plt.savefig("kat_1dgroup.png")
    
    