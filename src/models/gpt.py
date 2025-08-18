import torch
import torch.nn as nn
from torch.nn import functional as F
from .transformer_blocks import Block

# default params are configured for the GPT2 124M model
class Config:
    def __init__(self, vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight # tie weights
        self.apply(self._init_weights)  # initialize weights

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch_size, seq_length = idx.size()
        assert seq_length <= self.config.block_size, "sequence length exceeds block size"

        # forward the token and pos embeddings
        position = torch.arange(0, seq_length, dtype=torch.long, device=idx.device)
        position_embeddings = self.transformer.wpe(position)
        token_embeddings = self.transformer.wte(idx)
        x = token_embeddings + position_embeddings

        # forward through the blocks
        for block in self.transformer.h:
            x = block(x)

        # forward thru the final layer norm & classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # compute the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        # load a pretrained model from huggingface
        assert model_name_or_path in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], "model not supported"

        from transformers import GPT2LMHeadModel
        print(f"loading pretrained model {model_name_or_path} from huggingface...")

        configs = {
            'gpt2': dict(n_embd=768, n_layer=12, n_head=12), # 124M
            'gpt2-medium': dict(n_embd=1024, n_layer=24, n_head=16), # 350M
            'gpt2-large': dict(n_embd=1280, n_layer=36, n_head=20), # 774M
            'gpt2-xl': dict(n_embd=1600, n_layer=48, n_head=25) # 1558M
        }[model_name_or_path]

        configs['vocab_size'] = 50257 # these are the same for all models
        configs['block_size'] = 1024

        config = Config(**configs)
        model = GPT(config)

        state_dict = model.state_dict()
        state_dict_keys = state_dict.keys()
        state_dict_keys = [k for k in state_dict_keys if not k.endswith('.attn.bias')] # discard attention bias (mask)

        model_hf = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        state_dict_hf = model_hf.state_dict()

        state_dict_keys_hf = state_dict_hf.keys()
        state_dict_keys_hf = [k for k in state_dict_keys_hf if not k.endswith('.attn.masked_bias')] # discard masked bias
        state_dict_keys_hf = [k for k in state_dict_keys_hf if not k.endswith('.attn.bias')] # same


        assert len(state_dict_keys) == len(state_dict_keys_hf), "state dict keys mismatch"

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for key in state_dict_keys_hf:
            if any(key.endswith(w) for w in transposed):
                assert state_dict[key].shape == state_dict_hf[key].shape[::-1], f"shape mismatch for {key}"
                with torch.no_grad():
                    state_dict[key].copy_(state_dict_hf[key].t())

            else:
                assert state_dict[key].shape == state_dict_hf[key].shape, f"shape mismatch for {key}"
                with torch.no_grad():
                    state_dict[key].copy_(state_dict_hf[key])

        return model