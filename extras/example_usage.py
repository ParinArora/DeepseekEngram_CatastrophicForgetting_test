"""Small construction demo for the packaged models.

The goal of this file is to show the minimum amount of code needed to assemble
every model family and wrap it in the continual-learning adapter. It is a quick
sanity-check script, not a benchmark.
"""

from cl_models import (
    TransformerConfig,
    EngramConfig,
    TrainingConfig,
    StandardTransformerLM,
    TransformerWithEngramLM,
    ReducedAttentionTransformerLM,
    MLPOnlyLM,
    TorchCLModelAdapter,
)


def build_all_models():
    # Shared backbone config used across all four model families so the printed
    # parameter counts are easy to compare.
    base_cfg = TransformerConfig(
        vocab_size=50_000,
        max_seq_len=256,
        d_model=256,
        n_heads=8,
        n_layers=6,
        mlp_ratio=4,
        dropout=0.1,
        pad_token_id=0,
    )

    engram_cfg = EngramConfig(
        # This demo uses GPT-2 because it is widely available; experiment
        # scripts can swap in a different tokenizer without changing the API.
        tokenizer_name_or_path="gpt2",  # swap to a DeepSeek tokenizer if you want to mirror the official demo more closely
        engram_vocab_size=[64_000, 64_000],
        max_ngram_size=3,
        n_embed_per_ngram=128,
        n_head_per_ngram=4,
        layer_ids=[1, 4],
        pad_id=0,
    )

    train_cfg = TrainingConfig(device="cpu", epochs=1)

    # Each model is immediately wrapped so the return value matches what the
    # tester expects during actual continual-learning runs.
    baseline = TorchCLModelAdapter(StandardTransformerLM(base_cfg), train_cfg)
    engram = TorchCLModelAdapter(TransformerWithEngramLM(base_cfg, engram_cfg), train_cfg)
    reduced_attn = TorchCLModelAdapter(ReducedAttentionTransformerLM(base_cfg, attention_every_n_layers=2), train_cfg)
    mlp_only = TorchCLModelAdapter(MLPOnlyLM(base_cfg), train_cfg)

    return {
        "baseline": baseline,
        "engram": engram,
        "reduced_attention": reduced_attn,
        "mlp_only": mlp_only,
    }


if __name__ == "__main__":
    models = build_all_models()
    for name, adapter in models.items():
        print(name, adapter.num_parameters(), adapter.num_trainable_parameters())
