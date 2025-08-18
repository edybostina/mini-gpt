import torch
import torch.nn as nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # GPT2 used an approx. for GELU, more here https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)  

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.c_proj.SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # causal mask, or "bias" xD
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        batch_size, seq_length, nr_channels = x.size()

        # compute query, key, value 
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        k = k.view(batch_size, seq_length, self.n_head, nr_channels // self.n_head).transpose(1, 2)
        q = q.view(batch_size, seq_length, self.n_head, nr_channels // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_head, nr_channels // self.n_head).transpose(1, 2)

        # attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # attention = attention.masked_fill(self.bias[:,:,:seq_length,:seq_length] == 0, float('-inf'))
        # attention = F.softmax(attention, dim=-1)
        # # attention dropout
        # y = attention @ v

        # flash attention (replaces the 4 lines above)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, nr_channels)

        # output projection
        return self.c_proj(y)
    