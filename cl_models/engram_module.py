"""Implementation of the Engram side-memory used by the hybrid Transformer.

The package version keeps the same high-level ideas as the official Engram demo
while adapting the interfaces to fit the rest of this repo:

- token ids can optionally be compressed before hashing
- hashed n-gram ids are embedded and gated against backbone hidden states
- outputs are returned in the plain `[B, T, D]` residual-stream format expected
  by the rest of the models here

This is the most specialized file in the project. The rest of the package can
be read as ordinary Transformer code, while this file explains how Engram
creates an extra learned signal from hashed token contexts.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence
import math

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer
from tokenizers import normalizers, Regex

from .configs import EngramConfig


# -----------------------------------------------------------------------------
# Small math helpers
# -----------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """Small prime checker used for prime-sized hash tables."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    limit = int(n ** 0.5) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True


def next_prime(start: int, seen: set[int]) -> int:
    """Find the next unseen prime strictly larger than `start`."""
    candidate = max(2, start + 1)
    while True:
        if is_prime(candidate) and candidate not in seen:
            return candidate
        candidate += 1


# -----------------------------------------------------------------------------
# Tokenizer compression
# -----------------------------------------------------------------------------

class CompressedTokenizer:
    """
    Compress the raw tokenizer vocabulary into a canonical ID space.

    This follows the same normalization flow as the official DeepSeek Engram
    demo: Unicode normalization, accent stripping, lowercasing, whitespace
    normalization, and token-form merging.
    """

    def __init__(self, tokenizer_name_or_path: str) -> None:
        # The Hugging Face tokenizer supplies the original vocabulary that will
        # later be merged into a smaller canonical id space.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        sentinel = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ])
        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self) -> int:
        # This mirrors tokenizer APIs that report vocabulary size with `len(...)`.
        return self.num_new_token

    def _build_lookup_table(self) -> tuple[np.ndarray, int]:
        # `old2new` stores the final dense id remap, while `key2new` guarantees
        # that tokens with the same normalized surface form collapse together.
        old2new: Dict[int, int] = {}
        key2new: Dict[str, int] = {}
        new_tokens: List[str] = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            # Decode one token id at a time so the normalization happens at the
            # token level rather than on full text strings.
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            if "�" in text:
                # Replacement characters usually mean decoding is lossy, so fall
                # back to the tokenizer's raw token string for stability.
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]
        # `lookup` has length equal to the original vocabulary size; each entry
        # tells us which compressed id to use instead.
        return lookup, len(new_tokens)

    def _compress(self, input_ids: np.ndarray) -> np.ndarray:
        arr = np.asarray(input_ids, dtype=np.int64)
        # Negative ids are typically sentinels such as ignore-index values, so
        # they are left untouched.
        pos_mask = arr >= 0
        out = arr.copy()
        out[pos_mask] = self.lookup_table[arr[pos_mask]]
        return out

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        return self._compress(input_ids)


# -----------------------------------------------------------------------------
# ShortConv and hash mapping
# -----------------------------------------------------------------------------

class ShortConv(nn.Module):
    """
    Depthwise short convolution over a branch-expanded hidden tensor.

    This mirrors the official demo more closely than the earlier single-stream
    version:
      input  -> [B, T, HC_MULT, D]
      output -> [B, T, HC_MULT, D]

    Each branch gets its own RMSNorm before all branches are concatenated across
    channels and passed through one grouped 1D convolution.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        # Collapsing the branch dimension into the channel dimension lets one
        # depthwise Conv1d process every branch independently.
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps)
            for _ in range(hc_mult)
        ])
        self.act_fn = nn.SiLU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, HC_MULT, D]
        Returns:
            [B, T, HC_MULT, D]
        """
        bsz, seq_len, groups, hidden = x.shape
        if groups != self.hc_mult:
            raise ValueError(f"Input groups {groups} != hc_mult {self.hc_mult}")

        normed_chunks = []
        for i in range(groups):
            # Each branch gets its own normalization, matching the Engram idea
            # that branches may learn distinct gating dynamics.
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        x_norm = torch.cat(normed_chunks, dim=-1)  # [B, T, HC_MULT * D]

        # Conv1d expects channel-first data, so sequence and channel axes are
        # temporarily swapped.
        # `[B, T, HC_MULT * D] -> [B, HC_MULT * D, T]`
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        # Padding adds right-edge context that needs to be trimmed back off.
        y_bct = y_bct[..., :seq_len]
        y_bct = self.act_fn(y_bct)
        # Undo the earlier reshape so the output matches the input layout.
        # `[B, HC_MULT * D, T] -> [B, T, HC_MULT, D]`
        y = y_bct.transpose(1, 2).reshape(bsz, seq_len, groups, hidden).contiguous()
        return y


class NgramHashMapping:
    """
    Multi-head hashed n-gram mapping.

    This now follows the official demo structure very closely:
    - compressed tokenizer
    - per-layer odd multipliers
    - per-layer/per-ngram/per-head prime-sized tables
    - deterministic n-gram hashing
    """

    def __init__(self, cfg: EngramConfig) -> None:
        self.cfg = cfg
        self.vocab_size_per_ngram = cfg.engram_vocab_size
        self.max_ngram_size = cfg.max_ngram_size
        self.n_embed_per_ngram = cfg.n_embed_per_ngram
        self.n_head_per_ngram = cfg.n_head_per_ngram
        self.pad_id = cfg.pad_id
        self.layer_ids = cfg.layer_ids

        tokenizer_factory: Callable[[str], Any] = cfg.compressed_tokenizer_factory or CompressedTokenizer
        self.compressed_tokenizer = tokenizer_factory(cfg.tokenizer_name_or_path)
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            # Hashing operates in compressed-token space, so padding must be
            # remapped into that same space as well.
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])
        else:
            self.pad_id = 0

        # Multiplier selection is bounded to avoid overflowing int64 during the
        # token-mixing arithmetic used by the hash function.
        max_long = np.iinfo(np.int64).max
        m_max = int(max_long // max(1, self.tokenizer_vocab_size))
        half_bound = max(1, m_max // 2)

        prime_1 = 10007
        self.layer_multipliers: Dict[int, np.ndarray] = {}
        for layer_id in self.layer_ids:
            # Each layer gets its own deterministic multiplier set so the same
            # token context hashes differently at different insertion points.
            base_seed = int(cfg.seed + prime_1 * int(layer_id))
            rng = np.random.default_rng(base_seed)
            r = rng.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64,
            )
            self.layer_multipliers[layer_id] = r * 2 + 1

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self) -> Dict[int, List[List[int]]]:
        # Return format:
        #   layer_id -> list over n-gram order -> list over hash head
        # so callers can ask for the exact vocabulary size of any head.
        seen_primes: set[int] = set()
        vocab_size_across_layers: Dict[int, List[List[int]]] = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes: List[List[int]] = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_head_sizes: List[int] = []
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                current_prime_search_start = vocab_size - 1
                for _ in range(self.n_head_per_ngram):
                    # Distinct prime moduli per head reduce the chance that all
                    # heads collide on the same contexts.
                    found_prime = next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_head_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                all_ngram_vocab_sizes.append(current_ngram_head_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
        return vocab_size_across_layers

    def _get_ngram_hashes(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        # `input_ids` is `[B, T]` and the returned array is `[B, T, H]`, where
        # `H` is the total number of hash heads across all n-gram orders.
        x = np.asarray(input_ids, dtype=np.int64)
        bsz, seq_len = x.shape
        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x
            # Right-shifting with left padding creates aligned token histories
            # so position t can build hashes from its preceding context.
            shifted = np.pad(x, ((0, 0), (k, 0)), mode="constant", constant_values=self.pad_id)[:, :seq_len]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
        all_hashes: List[np.ndarray] = []

        for n in range(2, self.max_ngram_size + 1):
            ngram_index = n - 2
            tokens = base_shifts[:n]
            # Start from the oldest token in the window, then fold in the rest
            # with XOR to build an order-sensitive mixed integer.
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])

            head_vocab_sizes = self.vocab_size_across_layers[layer_id][ngram_index]
            for j in range(self.n_head_per_ngram):
                mod = int(head_vocab_sizes[j])
                # Each head uses a different modulus, producing parallel hashed
                # views of the same n-gram context.
                all_hashes.append((mix % mod).astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)  # [B, T, total_heads]

    def hash(self, input_ids: np.ndarray) -> Dict[int, np.ndarray]:
        # Compression happens once before all per-layer hashes are computed.
        input_ids = self.compressed_tokenizer(input_ids)
        return {
            layer_id: self._get_ngram_hashes(input_ids, layer_id=layer_id)
            for layer_id in self.layer_ids
        }


class MultiHeadEmbedding(nn.Module):
    """
    Shared embedding table with per-head offsets.

    This matches the official demo's dataflow closely.
    """

    def __init__(self, list_of_sizes: Sequence[int], dim_per_head: int) -> None:
        super().__init__()
        offsets = [0]
        for n in list_of_sizes[:-1]:
            offsets.append(offsets[-1] + n)
        # Offsets let each logical head address its own slice inside one shared
        # embedding table.
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.embedding = nn.Embedding(sum(list_of_sizes), dim_per_head)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # If `input_ids` is `[B, T, H]`, broadcasting `offsets` over the last
        # dimension keeps the same shape while shifting each head into its own
        # region of the shared embedding table.
        shifted_input_ids = input_ids + self.offsets
        # The embedding lookup then appends one embedding dimension:
        # `[B, T, H] -> [B, T, H, D_head]`
        return self.embedding(shifted_input_ids)


# -----------------------------------------------------------------------------
# Public Engram memory module used by the rest of the package
# -----------------------------------------------------------------------------

class EngramMemoryModule(nn.Module):
    """
    Engram memory module with an interface compatible with the rest of this
    package, while internally tracking the official demo more closely.

    External contract (kept unchanged):
      hidden_states: [B, T, D]
      input_ids:     [B, T]
      returns:       [B, T, D]

    Internal behavior:
    - expands hidden states to a demo-like branch dimension [B, T, HC_MULT, D]
    - applies per-branch key/query gating like the official demo
    - applies the official-style branchwise ShortConv
    - reduces back to one stream by averaging branches

    The branch dimension here is a compatibility shim: the original demo uses a
    mocked hyper-connection layout, but the rest of your package expects a plain
    Transformer hidden stream.
    """

    def __init__(self, d_model: int, layer_id: int, cfg: EngramConfig) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.cfg = cfg
        self.hidden_size = d_model

        # Reuse the existing config field as a branch-count control so we do not
        # have to change public exports or other files.
        self.hc_mult = max(1, int(getattr(cfg, "gating_hidden_multiplier", 1)))

        # Hashing + embedding turn discrete token contexts into dense Engram
        # features that can be compared to the current hidden state.
        self.hash_mapping = NgramHashMapping(cfg)
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_sizes=[x for y in self.hash_mapping.vocab_size_across_layers[layer_id] for x in y],
            dim_per_head=cfg.n_embed_per_ngram // cfg.n_head_per_ngram,
        )

        self.short_conv = ShortConv(
            hidden_size=d_model,
            kernel_size=cfg.kernel_size,
            dilation=cfg.max_ngram_size,
            hc_mult=self.hc_mult,
        )

        # Concatenating all hashed n-gram embeddings produces one Engram feature
        # vector of this width at each sequence position.
        engram_hidden_size = (cfg.max_ngram_size - 1) * cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, d_model)
        self.key_projs = nn.ModuleList([
            nn.Linear(engram_hidden_size, d_model)
            for _ in range(self.hc_mult)
        ])
        self.norm1 = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(self.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(self.hc_mult)])

    def _expand_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Expand [B, T, D] -> [B, T, HC_MULT, D] to mimic the demo's branch layout.
        """
        return hidden_states.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()

    def _reduce_hidden_states(self, branch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce [B, T, HC_MULT, D] back to [B, T, D].

        Averaging is the least opinionated reduction and keeps output scale stable.
        """
        return branch_tensor.mean(dim=2)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # Convert to a demo-like branch layout internally.
        # `[B, T, D] -> [B, T, HC_MULT, D]`
        branch_hidden_states = self._expand_hidden_states(hidden_states)

        # Hashing is performed through the compressed tokenizer and deterministic
        # per-layer n-gram hashing, just like the official demo.
        hash_ids = self.hash_mapping.hash(input_ids.detach().cpu().numpy())[self.layer_id]
        # `hash_ids` is a NumPy array `[B, T, H]`; converting to torch keeps the
        # same integer layout so the embedding lookup can use it.
        hash_input_ids = torch.from_numpy(hash_ids).to(device=hidden_states.device, dtype=torch.long)

        # [B, T, total_hash_heads, dim_per_head] -> [B, T, engram_hidden_size]
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)

        gates = []
        for hc_idx in range(self.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)

            query = branch_hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)

            # Keys come from Engram features; queries come from the current
            # backbone state. Their similarity decides how strongly Engram
            # should influence this branch at this position.
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)

        gates_t = torch.stack(gates, dim=2)  # [B, T, HC_MULT, 1]
        # The same Engram value features are shared across branches, but each
        # branch gets its own learned gate.
        value = gates_t * self.value_proj(embeddings).unsqueeze(2)  # [B, T, HC_MULT, D]
        output = value + self.short_conv(value)

        # Return to the package's public single-stream contract.
        # `[B, T, HC_MULT, D] -> [B, T, D]`
        return self._reduce_hidden_states(output)
