import math
import torch
from torch import nn

import bitsandbytes as bnb


class LoRAModule(nn.Module):
    def __init__(
        self,
        lora_name,
        original_module: nn.Module,
        multiplier=1.0,
        lora_dim=16,
        alpha=16,
        dropout=None,
        module_dropout=None,
        decompose=False
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.multiplier = multiplier
        self.original_module = original_module
        self.dropout = dropout
        self.module_dropout = module_dropout

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim

        self.dora_mag = None


    def init_weights(self):
        # Initialize up and down the established way
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        # Set dora magnitude to that of the original module weight
        if self.dora_mag is not None:
            self.dora_mag.weight.data = (torch.linalg.norm(self.original_module.weight.detach(), dim=1)).unsqueeze(1).detach()

    def forward(self, x):
        # Module dropout (skip lora module)
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.original_module(x)

        # Down to low-rank
        lx = self.lora_down(x)

        # Regular dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # Back up to full-rank
        lx = self.lora_up(lx)

        # Add scaled residual to original
        lx = self.original_module(x) + lx * self.scale * self.multiplier

        # Return regular lora result
        if self.dora_mag is None:
            return lx
        
        # Calculate V + dV for dora scaling in f32 to prevent overflow
        v_plus_dv = self.original_module.weight + (self.lora_up.weight @ self.lora_down.weight) * self.scale
        v_plus_dv_f32 = v_plus_dv.float()
        norm = torch.linalg.norm(v_plus_dv_f32, dim=1).detach()
        mag = self.dora_mag.weight.view(-1)
        norm_scale = (mag / (norm + 1e-6)).to(x.dtype)
        
        # Apply scaling to the already computed lx = x(V + dV)
        return norm_scale * lx

    def inject(self, parent_module):
        # Replace original module with lora module
        parent_module._modules[self.lora_name.split("/")[-1]] = self

    def inject_forward(self):
        # Replace original module's forward method with lora forward
        self.original_forward = self.original_module.forward
        self.original_module.forward = self.forward

    def dump_weights(self):
        # Update original module weights
        dtype = self.original_module.weight.dtype
        
        if self.dora_mag is None:
            updated = self.original_module.weight + (self.lora_up.weight @ self.lora_down.weight) * self.scale
        else:
            # Apply DoRA weight update formula: W_new = m * (V + dV) / ||V + dV||
            # We perform the norm in float32 to prevent overflow in FP16
            v_plus_dv = self.original_module.weight + (self.lora_up.weight @ self.lora_down.weight) * self.scale
            mag = self.dora_mag.weight.view(-1, 1)
            v_plus_dv_f32 = v_plus_dv.float()
            norm = torch.linalg.norm(v_plus_dv_f32, dim=1, keepdim=True)
            updated = mag * (v_plus_dv_f32 / (norm + 1e-6))
            updated = updated.to(dtype)

        self.original_module.weight.data = updated.to(dtype).clone().detach()

        # Reinit lora weights
        self.init_weights()


class LoRALinear(LoRAModule):
    def __init__(
        self,
        lora_name,
        original_module: nn.Module,
        decompose,
        **kwargs
    ):
        super().__init__(
            lora_name,
            original_module,
            **kwargs
        )
        self.in_dim = original_module.in_features
        self.out_dim = original_module.out_features
        self.lora_dim = min(self.lora_dim, self.in_dim, self.out_dim)
        self.lora_down = torch.nn.Linear(self.in_dim, self.lora_dim, bias=False)
        self.lora_up = torch.nn.Linear(self.lora_dim, self.out_dim, bias=False)
        if decompose:
            self.dora_mag = torch.nn.Linear(1, self.out_dim)
    
        self.init_weights()

    def resize(self, lora_dim):
        self.lora_dim = lora_dim
        self.lora_down = torch.nn.Linear(self.in_dim, self.lora_dim, bias=False)
        self.lora_up = torch.nn.Linear(self.lora_dim, self.out_dim, bias=False)
        self.init_weights()
            
    def quantize(self):
        original_module_q = bnb.nn.Linear4bit(self.original_module.in_features, self.original_module.out_features, bias=self.original_module.bias is not None)
        original_module_q.load_state_dict(self.original_module.state_dict())
        self.original_module = original_module_q


class LoRAConv1d(LoRAModule):
    def __init__(
        self,
        lora_name,
        original_module: nn.Module,
        decompose,
        **kwargs
    ):
        super().__init__(
            lora_name,
            original_module,
            **kwargs
        )
        in_dim = original_module.in_channels
        out_dim = original_module.out_channels
        kernel_size = original_module.kernel_size
        stride = original_module.stride
        padding = original_module.padding
        self.lora_down = torch.nn.Conv1d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
        self.lora_up = torch.nn.Conv1d(self.lora_dim, out_dim, 1, 1, bias=False)
        if decompose:
            self.dora_mag = torch.nn.Linear(1, self.out_dim)
    
        self.init_weights()

    def resize(self, lora_dim):
        return

    def quantize(self):
        return
