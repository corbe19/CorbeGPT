from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
import math


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3) # Linear layer for query, key, value, not copying vector three times,

        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # Linear layer for output projection

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size)) # storing mask in buffer

    def forward(self, x): # x = input tensor/batch of tokens
        B, T, C = x.size() # Batch size, sequence length, embedding dimension

        qkv = self.c_attn(x) # Apply linear layer
        q, k, v = qkv.split(self.n_embd, dim=2) # Split into query, key, value
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # Scaled dot-product attention
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # Apply causal mask, inf + softmax = 0
        att = F.softmax(att, dim=-1) # Turns logits into probabilities

        y = att @ v # Apply attention to value
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y) # Project back to original dimension
        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4) # First linear layer, expand
        self.gelu = nn.GELU(approximate='tanh') # Activation function, 
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd) # Second linear layer

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # Communicate
        x = x + self.mlp(self.ln2(x)) # Think
        return x

@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Weights of token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # Weights of positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd), # Tidy everything up
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Logits