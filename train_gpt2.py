"""
GPT-2 Training Script - Reproducing GPT-2 (124M) from scratch
Based on Andrej Karpathy's "Let's reproduce GPT-2" video lecture

=== OVERVIEW ===
This script trains a GPT-2 124M parameter model from scratch on the FineWeb-Edu dataset.
The goal is to match or exceed the original GPT-2's performance on HellaSwag benchmark
(~29.5% accuracy for 124M model) in a reasonable amount of compute time.

=== MODEL ARCHITECTURE ===
GPT-2 is a decoder-only transformer with the following structure:
- Token + Position Embeddings → N Transformer Blocks → LayerNorm → LM Head
- Each block: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual
- Pre-norm architecture (LayerNorm before attention/MLP, not after)

Key hyperparameters for GPT-2 124M:
- n_layer: 12 transformer blocks
- n_head: 12 attention heads  
- n_embd: 768 embedding dimension
- vocab_size: 50257 (GPT-2's BPE vocabulary) → padded to 50304 for GPU efficiency
- block_size: 1024 token context window

=== TRAINING OPTIMIZATIONS ===

1. MIXED PRECISION (bfloat16):
   - Forward pass uses bfloat16 via torch.autocast for 2x memory savings
   - Gradients remain in float32 for numerical stability
   - No gradient scaling needed (unlike fp16)

2. FLASH ATTENTION:
   - Uses PyTorch's scaled_dot_product_attention with Flash Attention kernel
   - O(N) memory instead of O(N²), much faster for long sequences
   - Fuses attention computation into a single GPU kernel

3. TORCH.COMPILE:
   - JIT compiles the model, fusing operations and reducing kernel launches
   - ~2x speedup on modern GPUs (especially with mode="reduce-overhead")

4. FUSED ADAMW:
   - Uses CUDA-fused AdamW optimizer when available
   - Single kernel for all optimizer operations instead of multiple

5. GRADIENT ACCUMULATION:
   - Simulates large batch sizes (524,288 tokens = 2^19) with smaller micro-batches
   - Accumulates gradients over multiple forward/backward passes before optimizer step
   - Enables training with large effective batch sizes on limited GPU memory

6. VOCAB SIZE PADDING:
   - Original vocab: 50257 → Padded to 50304 (divisible by 128)
   - Better memory alignment and GPU utilization

=== DISTRIBUTED DATA PARALLEL (DDP) ===
Supports multi-GPU training via torchrun:
  $ torchrun --standalone --nproc_per_node=8 train_gpt2.py

DDP splits data across GPUs, each processes different batches:
- Each GPU maintains a full model copy
- Gradients are averaged across GPUs via all-reduce after backward pass
- require_backward_grad_sync optimization skips sync during accumulation steps

Key DDP variables:
- ddp_rank: Global rank of this process (0 to world_size-1)
- ddp_local_rank: Rank within this node (for device assignment)
- ddp_world_size: Total number of processes/GPUs
- master_process: Only rank 0 does logging/checkpointing

=== LEARNING RATE SCHEDULE ===
Cosine decay with linear warmup (following GPT-3 paper):
- Warmup: Linear increase from 0 to max_lr over first 375M tokens
- Decay: Cosine decay from max_lr to min_lr (0.1 * max_lr)
- max_lr = 6e-4, min_lr = 6e-5

=== WEIGHT DECAY ===
Selective weight decay (following GPT-2/3):
- 0.1 weight decay on 2D parameters (weights in Linear layers)
- No weight decay on 1D parameters (biases, LayerNorm scales/shifts)
- Decoupled weight decay (AdamW style, not L2 regularization)

=== DATA LOADING ===
DataLoaderLite loads pre-tokenized FineWeb-Edu shards:
- Shards created by fineweb.py (~100M tokens each, stored as .npy)
- Each GPU reads different portions (process_rank offset)
- Automatically advances to next shard when current is exhausted
- Validation shard separate from training shards

=== INITIALIZATION ===
Careful weight initialization for stable training:
- All weights: Normal(0, 0.02)
- Residual projections (c_proj): Scaled by 1/sqrt(2*n_layer)
- This prevents residual stream variance from exploding with depth

=== WEIGHT TYING ===
Token embeddings (wte) are shared with the output projection (lm_head):
- Reduces parameters by n_vocab * n_embd (~38M for GPT-2)
- Empirically works well and is standard practice

=== FILES ===
- train_gpt2.py: This training script
- fineweb.py: Dataset download and tokenization
- edu_fineweb10B/: Directory containing tokenized shards
- input.txt: Small Shakespeare dataset for quick testing
"""

from dataclasses import dataclass
import inspect
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

#----------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention (materializes the large (T,T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # use the new PyTorch scaled_dot_product_attention function - flash attention implementation
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class TanhGELU(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # wte  = word token embedding
        # wpe  = word position embedding
        # h    = hidden
        # ln_f = layer norm final
        # module that allows you to store modules in a dictionary
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight  # weight tying 30%
        # this means that the token embedding weights and the language modeling head weights are the same
        # this reduces the number of parameters by 30% and improves performance
        # the lm head represents the final linear layer that maps the hidden states to the vocabulary logits
        # reason we had to tie them is because out own GPT implementation does not use a Conv1D layer like the original GPT-2 model
        # a Conv1D layer is just a linear layer with some special weight initialization
        # we didnt use it because it is not in the PyTorch standard library

        self.apply(self._init_weights)

    # Initialize weights as in the original GPT paper (https://arxiv.org/abs/2005.14165)
    # Language models are unsupervised multitask learners paper
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        """ Loads pretrained GPT-2 model from huggingface transformers 
            This does: 
            1) create a from-scratch initialized mingpt model 
            2) load weights from huggingface transformers GPT-2 model into it
            3) return mingpt model
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}, "Only gpt2 model is supported"
        from transformers import GPT2LMHeadModel
        print("Loading pretrained model from huggingface transformers: %s", model_type)

        # n_layer, n_head, n_embd based on model type
        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for gpt2 models
        config_args['block_size'] = 1024 # always 1024 for gpt2 models

        # create a from scratch initialized model mingpt
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = set(sd.keys())
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # discard this mask / buffer

        # init a huggingface transformer model

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict() # sd stands for state dict that shows the weights and biases of the model

        # copy weights from hf model to mingpt model
        sd_keys_hf = set(sd_hf.keys())
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")] # discard this mask / buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")] # same, just the masked
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# -----------------------------------------------------------------------------

import tiktoken
import numpy as np

enc = tiktoken.get_encoding('gpt2')

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, master_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = []
        if os.path.exists(data_root):
            shards = os.listdir(data_root)
            shards = [s for s in shards if split in s]
            shards = sorted(shards)
            shards = [os.path.join(data_root, s) for s in shards]
        
        if len(shards) > 0:
            # use FineWeb shards
            self.shards = shards
            if master_process:
                print(f"found {len(shards)} shards for split {split}")
            # state, init at shard zero
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.use_shards = True
        else:
            # fallback to input.txt for testing
            if master_process:
                print(f"no shards found, falling back to input.txt")
            with open('input.txt', 'r') as f:
                text = f.read()
            enc = tiktoken.get_encoding('gpt2')
            tokens = enc.encode(text)
            self.tokens = torch.tensor(tokens, dtype=torch.long)
            self.shards = None
            self.current_shard = 0
            self.use_shards = False
            if master_process:
                print(f"loaded {len(self.tokens)} tokens from input.txt")

        self.current_position = self.B * self.T * self.process_rank

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        if self.use_shards:
            self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = buf.to(device)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor (skip over other processes' data)
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            if self.use_shards:
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# Distributed Data Parallel (DDP) Explained:
# -------------------------------------------
# DDP allows training across multiple GPUs (on one or more machines).
# Each GPU runs its own process with a copy of the model.
#
# How it works:
# 1. Each process loads the same model and processes DIFFERENT data batches
# 2. Each process computes gradients independently (forward + backward)
# 3. Before optimizer.step(), DDP automatically ALL-REDUCES gradients across all GPUs
#    (averages them so all processes have identical gradients)
# 4. Each process updates its model identically → models stay in sync
#
# Key environment variables (set by torchrun):
# - RANK: Global process ID (0, 1, 2, ... across all machines)
# - LOCAL_RANK: GPU ID on this machine (0, 1, 2, ... per machine)
# - WORLD_SIZE: Total number of processes/GPUs
#
# Why divide grad_accum_steps by world_size?
# - With 8 GPUs, each processes B*T tokens per micro-step
# - Total tokens per micro-step = B*T*8 (8x more than single GPU!)
# - So we need 8x fewer accumulation steps to reach same total_batch_size
#
# Run with: torchrun --nproc_per_node=NUM_GPUS train_gpt2.py

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
print(f"using device: {device}")

import time

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Batch Size Calculation with DDP:
# ----------------------------------
# total_batch_size = 524288 tokens (what we want per optimizer step)
# Each GPU processes: B * T = 16 * 1024 = 16384 tokens per micro-step
# With ddp_world_size GPUs: 16384 * world_size tokens per micro-step across all GPUs
# grad_accum_steps = total_batch_size / (B * T * world_size)
#
# Example with 8 GPUs:
#   grad_accum_steps = 524288 / (16 * 1024 * 8) = 524288 / 131072 = 4
#   Each step: 8 GPUs × 4 accum × 16384 tokens = 524288 tokens ✓
#
# Example with 1 GPU:
#   grad_accum_steps = 524288 / (16 * 1024 * 1) = 524288 / 16384 = 32
#   Each step: 1 GPU × 32 accum × 16384 tokens = 524288 tokens ✓

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 16 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", master_process=master_process)

# set matmul precision to high for better performance with bfloat16 tf32
torch.set_float32_matmul_precision('high') 

# gradient scaling if fp16 sao we use bf16

# get logits
model = GPT(GPTConfig(vocab_size=50304))  # round up vocab_size to nearest multiple of 128 for GPU efficiency
model.to(device)
# model = torch.compile(model) # requires Triton, not available on Windows
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_steps steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > max_steps, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)

# Gradient Accumulation Explained:
# ---------------------------------
# We want to train with a large batch size (524288 tokens) for better gradient estimates,
# but we can't fit that many tokens in GPU memory at once.
# Solution: Process smaller "micro-batches" and accumulate gradients over multiple forward/backward passes.
#
# Example: total_batch_size=524288, micro_batch=16*1024=16384 tokens
#          grad_accum_steps = 524288 / 16384 = 32 micro-batches per optimizer step
#
# Why scale loss by 1/grad_accum_steps?
# - Each backward() call ADDS gradients to .grad (PyTorch default behavior)
# - After 32 backward() calls, gradients are 32x larger than a single batch
# - But we want the MEAN gradient (as if we processed one big batch)
# - So we divide loss by 32 BEFORE backward(), making gradients 32x smaller
# - Result: accumulated gradients = mean over all 32 micro-batches ✓

for step in range(max_steps):
    t0 = time.time()

    # once in a while evaluate our validation loss
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")

    # once in a while generate from the model (except step 0, which is noise)
    if (step > 0 and step % 100 == 0) or (step == max_steps - 1):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # Only sync gradients on the last micro-step (DDP optimization)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    # Average loss across all GPUs for consistent logging
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # clip the gradient norm to 1.0 to prevent exploding gradients when using fp16 / bf16
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # set learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)

# prefix tokens
model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)


""""
Expected output: A loss around 10-11 (since 
−ln⁡(1/50257)≈10.82
−ln(1/50257)≈10.82 for random uniform predictions over vocab).
"""