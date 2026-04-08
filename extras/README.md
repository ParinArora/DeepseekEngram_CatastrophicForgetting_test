# Continual Learning Architecture Pack

This package gives you a **shared experimental scaffold** for comparing multiple
causal-LM architectures under the same continual-learning tester.

Included architectures:

- `StandardTransformerLM` — baseline Transformer
- `TransformerWithEngramLM` — Transformer + Engram memory injection
- `ReducedAttentionTransformerLM` — attention weakened by keeping only every N-th attention layer
- `MLPOnlyLM` — no-attention ablation backbone

Important note on Engram:

- The official DeepSeek Engram GitHub repo currently exposes a **demo module**, not a full production training stack.
- The Engram implementation here is a **clean integration inspired by the official demo design**:
  - compressed tokenizer
  - hashed n-gram lookup
  - multi-head memory tables
  - context-aware gating
  - local short convolution refinement
- This makes it practical to compare `baseline -> +Engram` inside one shared trainer.

## Expected task data format

The adapter/trainer expects PyTorch-style dataloaders producing batches like:

```python
{
    "input_ids": LongTensor[B, T],
    "labels": LongTensor[B, T],
}
```

## Suggested workflow

1. Build task dataloaders per continual-learning stage.
2. Instantiate one of the model classes.
3. Wrap it in `TorchCLModelAdapter`.
4. Plug that into the tester we built earlier.
5. Compare architectures under the same metric/probe suite.

## Recommended comparison grid

- `StandardTransformerLM`
- `TransformerWithEngramLM`
- `ReducedAttentionTransformerLM`
- `ReducedAttentionTransformerLM + Engram` (easy next file to add)
- `MLPOnlyLM`
- `MLPOnlyLM + Engram` (easy next file to add)

Right now this artifact gives you the **first batch** of backbone files and the shared trainer/adapter layer.
