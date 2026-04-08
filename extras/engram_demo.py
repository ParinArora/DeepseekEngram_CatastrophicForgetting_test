"""
================================================================================
[Engram Architecture Demo Implementation]

DISCLAIMER:
1. Demo Purpose Only:
   This code is a demonstration version intended to illustrate the core logic and
   data flow of the Engram module. In this repository it serves as a reference
   script for understanding the packaged implementation in `cl_models`.

2. Production Readiness:
   This implementation requires further optimization for actual production use
   (e.g., custom CUDA kernels, distributed training support).

3. Simplifications:
   Standard components (Normalization, Attention, MoE) and complex Hyper-connection
   mechanisms are omitted or mocked in this version to focus exclusively on the
   Engram module implementation.
================================================================================
"""

"""
pip install torch numpy transformers sympy
"""

## built-in
from typing import List
from dataclasses import dataclass, field
import math

## third-party
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex


# ------------------------------------------------------------------------------
# Configuration for the Engram module.
# This holds all hyperparameters that control:
# - tokenizer choice
# - hashed n-gram vocabulary sizes
# - number of layers where Engram is active
# - embedding sizes
# - convolution kernel size
# ------------------------------------------------------------------------------
@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280 * 5, 129280 * 5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4


# ------------------------------------------------------------------------------
# Configuration for the backbone model (the surrounding transformer-like model).
# This demo uses a simplified backbone with:
# - hidden size
# - hyper-connection multiplicity
# - vocabulary size
# - number of layers
# ------------------------------------------------------------------------------
@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30


# Instantiate global configs used throughout the script.
engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()


# ------------------------------------------------------------------------------
# CompressedTokenizer
#
# Goal:
# Reduce tokenizer vocabulary by merging tokens that normalize to the same text.
#
# Why this exists:
# Some token IDs may decode to slightly different text forms that are effectively
# equivalent after normalization (case folding, accent stripping, whitespace cleanup).
# Compressing them helps make hashing more stable and potentially reduces collisions
# caused by superficial token form differences.
# ------------------------------------------------------------------------------
class CompressedTokenizer:
    def __init__(
            self,
            tokenizer_name_or_path,
    ):
        # Load the original tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

        # Private sentinel character used to preserve a single-space token during stripping.
        SENTINEL = "\uE000" #! temporary placeholder character used during text normalization so the code does not accidentally erase a token that is exactly one space.

        # Build a normalization pipeline:
        # - NFKC/NFD normalize Unicode
        # - strip accents
        # - lowercase
        # - collapse whitespace
        # - preserve single-space edge case
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),  # ! Normalize compatibility Unicode forms into a standard composed form
            normalizers.NFD(),  # ! Decompose characters so accents become separate combining marks
            normalizers.StripAccents(),  # ! Remove accent marks, e.g. "é" -> "e"
            normalizers.Lowercase(),  # ! Convert all text to lowercase
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),  # ! Collapse any run of spaces/tabs/newlines into one space
            normalizers.Replace(Regex(r"^ $"), SENTINEL),   # ! If the whole string is exactly one space, protect it with a sentinel
            normalizers.Strip(),  # ! Remove leading and trailing spaces
            normalizers.Replace(SENTINEL, " "),  # ! Restore the protected single-space value
        ])

        # Build mapping from original token IDs -> compressed token IDs.
        self.lookup_table, self.num_new_token = self._build_lookup_table() #! cleans up the lookup table so different tokens like åmsterdam and amsterdam get mapped to the same token ID

    def __len__(self):
        # Return compressed vocabulary size.
        return self.num_new_token

    def _build_lookup_table(self):
        # old2new maps original token ID -> compressed token ID.
        old2new = {}

        # key2new ensures identical normalized strings share the same compressed ID.
        key2new = {}

        # new_tokens storess unique normalized token keys.
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            # Decode each token ID individually.
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            # If decoding contains replacement characters, fall back to raw token string.
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                # Normalize text so equivalent forms map together.
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            # Reuse compressed ID if this normalized key already exists.
            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        # Build a dense NumPy lookup array for fast remapping.
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)

    def _compress(self, input_ids):
        # Convert input token IDs to compressed token IDs.
        arr = np.asarray(input_ids, dtype=np.int64)

        # Only map non-negative IDs.
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out

    def __call__(self, input_ids):
        # Make the tokenizer callable.
        return self._compress(input_ids)


# ------------------------------------------------------------------------------
# ShortConv
#
# A depthwise 1D convolution module applied over sequence length.
#
# Input shape:  (B, L, HC_MULT, D)
# Output shape: (B, L, HC_MULT, D)
#
# Purpose:
# Add local temporal mixing to Engram-produced values.
# Each channel is convolved independently (depthwise/grouped conv), which is cheap
# and preserves channel structure.
# ------------------------------------------------------------------------------
class ShortConv(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            kernel_size: int = 4,
            dilation: int = 1,
            norm_eps: float = 1e-5,
            hc_mult: int = 4,
            activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation

        # Total channels after flattening hyper-connection groups.
        total_channels = hidden_size * hc_mult

        # Depthwise 1D convolution:
        # groups=total_channels means each channel is convolved separately.
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        # One RMSNorm per hyper-connection group.
        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps)
            for _ in range(hc_mult)
        ])

        # Optional activation after convolution.
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,L,HC_MULT,D)
        Output: (B,L,HC_MULT,D)
        """
        B, T, G, C = x.shape

        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        # Normalize each group independently.
        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))

        # Concatenate groups along the feature dimension:
        # shape becomes (B, T, G*C)
        x_norm = torch.cat(normed_chunks, dim=-1)

        # Conv1d expects (B, C, T), so transpose sequence and channel dims.
        x_bct = x_norm.transpose(1, 2)

        # Apply depthwise convolution over time.
        y_bct = self.conv(x_bct)

        # Trim to original sequence length because padding creates extra right context.
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)

        # Restore original 4D shape.
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()

        return y


# ------------------------------------------------------------------------------
# Utility: find the next prime number greater than "start" that is not already used.
#
# Why primes?
# Using prime moduli for hash buckets can improve distribution and help reduce
# pathological collision patterns.
# ------------------------------------------------------------------------------
def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


# ------------------------------------------------------------------------------
# NgramHashMapping
#
# Core role:
# Convert input token sequences into hashed n-gram IDs for each selected layer.
#
# Pipeline:
# 1. Compress tokenizer IDs
# 2. Build shifted token windows for n-grams
# 3. Mix tokens with layer-specific random multipliers
# 4. Hash into multiple heads with different prime moduli
#
# Output:
# For each layer, returns a tensor of shape (B, T, total_ngram_heads)
# where each position stores hashed bucket IDs.
# ------------------------------------------------------------------------------
class NgramHashMapping:
    def __init__(
            self,
            engram_vocab_size,
            max_ngram_size,
            n_embed_per_ngram,
            n_head_per_ngram,
            layer_ids,
            tokenizer_name_or_path,
            pad_id,
            seed,
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        # Build compressed tokenizer.
        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)

        # Remap pad ID into compressed-token space.
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        # Compute safe upper bound for multiplier generation to avoid int64 overflow.
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)

        # Prime used to vary RNG seed by layer.
        PRIME_1 = 10007

        # Per-layer random multipliers for each token position within the n-gram.
        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)

            # Random integers used to derive odd multipliers.
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        # Precompute per-layer, per-ngram, per-head hash vocabulary sizes (prime numbers).
        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        # Keep all primes unique across all layers/heads.
        seen_primes = set()
        vocab_size_across_layers = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []

                # Base desired vocab size for this n-gram order.
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1

                # Assign a unique prime modulus per head.
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start,
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(
            self,
            input_ids: np.ndarray,
            layer_id: int,
    ) -> np.ndarray:
        # Ensure array is int64 for safe hashing arithmetic.
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        # Layer-specific random multipliers.
        multipliers = self.layer_multipliers[layer_id]

        # shift_k(k) returns x shifted right by k positions, padding on the left.
        # This creates aligned previous-token contexts for n-gram construction.
        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)),
                             mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        # Precompute all shifted views up to max_ngram_size - 1.
        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []

        # For each n-gram size (2-gram, 3-gram, ...)
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]

            # Mix token IDs using multiplier-weighted XOR hashing.
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])

            # Multiple hash heads for the same n-gram size.
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])

                # Hash into bucket range for this head.
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        # Stack head hashes into final shape: (B, T, total_heads)
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        # First compress original tokenizer IDs.
        input_ids = self.compressed_tokenizer(input_ids)

        # Compute hashes for each requested layer.
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers


# ------------------------------------------------------------------------------
# MultiHeadEmbedding
#
# Stores one large embedding table but logically splits it into multiple head-
# specific vocabularies via offsets.
#
# Input shape:  (..., num_heads)
# Output shape: (..., num_heads, D)
#
# Why this design?
# Each hash head may have a different vocabulary size (different prime modulus).
# Offsets let all heads share one nn.Embedding while still indexing disjoint
# regions of the table.
# ------------------------------------------------------------------------------
class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        # Build prefix-sum offsets so each head occupies a separate slice of the table.
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        # Single shared table containing all head vocabularies back-to-back.
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Shift each head's IDs into its own offset range.
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids)

        return output


# ------------------------------------------------------------------------------
# Engram module
#
# This is the main feature extractor injected into selected backbone layers.
#
# High-level flow:
# 1. Hash token sequence into multi-head n-gram IDs
# 2. Embed those hash IDs
# 3. Build keys from embeddings and compare against current hidden states (queries)
# 4. Produce gates per hyper-connection branch
# 5. Project embedding features into values
# 6. Add short convolutional local mixing
#
# Output shape matches hidden_states: (B, L, HC_MULT, D)
# ------------------------------------------------------------------------------
class Engram(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        # Build hash mapping object for all selected Engram layers.
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size=engram_cfg.max_ngram_size,
            n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
            n_head_per_ngram=engram_cfg.n_head_per_ngram,
            layer_ids=engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed,
        )

        # Embedding table for all n-gram hash heads in this specific layer.
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=[x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D=engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )

        # Local temporal mixing applied to projected/gated values.
        self.short_conv = ShortConv(
            hidden_size=backbone_config.hidden_size,
            kernel_size=engram_cfg.kernel_size,
            dilation=engram_cfg.max_ngram_size,
            hc_mult=backbone_config.hc_mult,
        )

        # Total concatenated embedding size across all n-gram orders.
        engram_hidden_size = (engram_cfg.max_ngram_size - 1) * engram_cfg.n_embed_per_ngram

        # Shared value projection from Engram embedding space -> backbone hidden size.
        self.value_proj = nn.Linear(engram_hidden_size, backbone_config.hidden_size)

        # One key projection per hyper-connection branch.
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
        )

        # Separate normalization for keys and queries in each branch.
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])

    def forward(self, hidden_states, input_ids):
        """
        hidden_states: [B, L, HC_MULT, D]
        input_ids: [B, L]
        """
        # Hash token IDs for this layer, then convert NumPy -> torch tensor.
        hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id])

        # Embed each hash head, then flatten head and head-dim together.
        # Result shape roughly: [B, L, total_ngram_embedding_dim]
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)

        gates = []
        for hc_idx in range(backbone_config.hc_mult):
            # Project Engram embedding into a "key" for this hyper-connection branch.
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)

            # Use existing hidden state branch as the "query".
            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)

            # Similarity score between Engram feature and current branch hidden state.
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)

            # Nonlinear transformation:
            # - preserve sign
            # - compress magnitude
            # - avoid exact zero
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()

            # Squash to (0, 1) for gating.
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)

        # Stack branch gates into shape [B, L, HC_MULT, 1]
        gates = torch.stack(gates, dim=2)

        # Project Engram embeddings into values and gate per branch.
        value = gates * self.value_proj(embeddings).unsqueeze(2)

        # Add local convolutional enhancement.
        output = value + self.short_conv(value)
        return output


# ------------------------------------------------------------------------------
# TransformerBlock
#
# Simplified transformer block for the demo.
# Attention and MoE are mocked as identity functions.
# Engram is inserted only on selected layers.
# ------------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, layer_id):
        super().__init__()

        # Mock attention and MoE for demonstration.
        self.attn = lambda x: x
        self.moe = lambda x: x

        self.engram = None
        if layer_id in engram_cfg.layer_ids:
            self.engram = Engram(layer_id=layer_id)

    def forward(self, input_ids, hidden_states):
        # If this layer has an Engram module, apply it residually.
        if self.engram is not None:
            hidden_states = self.engram(hidden_states=hidden_states, input_ids=input_ids) + hidden_states

        # Mock attention residual.
        hidden_states = self.attn(hidden_states) + hidden_states

        # Mock MoE residual.
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states


# ------------------------------------------------------------------------------
# Demo entry point
#
# This constructs a toy LLM-like stack:
# - token embedding
# - 30 simplified transformer blocks
# - final linear head
#
# Then it runs a forward pass on one sample sentence.
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    LLM = [
        nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size),
        *[TransformerBlock(layer_id=layer_id) for layer_id in range(backbone_config.num_layers)],
        nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size)
    ]

    # Example input text.
    text = "Only Alexander the Great could tame the horse Bucephalus."

    # Original tokenizer used to generate token IDs.
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path, trust_remote_code=True)
    input_ids = tokenizer(text, return_tensors='pt').input_ids

    B, L = input_ids.shape

    for idx, layer in enumerate(LLM):
        if idx == 0:
            # Initial token embedding: [B, L, D]
            hidden_states = LLM[0](input_ids)

            ## mock hyper-connection
            # Expand into multiple hyper-connection branches:
            # [B, L, D] -> [B, L, HC_MULT, D]
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, backbone_config.hc_mult, -1)

        elif idx == len(LLM) - 1:
            ## mock hyper-connection
            # Collapse back to one branch before output projection.
            hidden_states = hidden_states[:, :, 0, :]
            output = layer(hidden_states)

        else:
            # Apply each simplified transformer block.
            hidden_states = layer(input_ids=input_ids, hidden_states=hidden_states)

    print("✅ Forward Complete!")
    print(f"{input_ids.shape=}\n{output.shape=}")
