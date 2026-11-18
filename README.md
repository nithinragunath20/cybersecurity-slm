---
license: mit
library_name: pytorch
tags:
- cybersecurity
- language-model
- transformer
- darpa-transparent-computing
- telemetry
---

# Cybersecurity SLM

An approximately 80.66-million-parameter decoder-only Transformer language model trained from scratch on structured cybersecurity telemetry based on DARPA Transparent Computing CDM18 data.

## Model Details

| Property | Value |
|---|---|
| Model type | Decoder-only Transformer |
| Parameters | 80,658,432 |
| Layers | 4 |
| Hidden dimension | 256 |
| Attention heads | 8 |
| KV groups | 2 |
| Feed-forward dimension | 768 |
| Maximum context length | 256 tokens |
| Vocabulary size | 151,643 |
| Training | From scratch |
| Tokenizer | Qwen3 tokenizer |

This is a custom Qwen3-inspired architecture. It does not use pretrained Qwen model weights.

## Dataset

Dataset: [nithinragunath/DT](https://huggingface.co/datasets/nithinragunath/DT)

The dataset contains 1,000,000 structured CDM18 telemetry records:

- Training: 700,000 records
- Validation: 150,000 records
- Test: 150,000 records

The records include fields such as record type, CDM version, source, and JSON event details.

## Training Configuration

- Optimizer: AdamW
- Learning rate: 0.0001
- Batch size: 2
- Gradient accumulation: 4
- Effective batch size: 8
- Sequence length: 256
- Training checkpoint selected: iteration 76,000

## Evaluation

Best checkpoint results:

| Metric | Value |
|---|---:|
| Training loss | 0.5774 |
| Validation loss | 0.5374 |
| Sampled test loss | 0.6253 |

The test loss was measured using 50 randomly sampled test batches.

## Intended Use

This model is an experimental research project for learning and generating continuations of structured cybersecurity telemetry.

It is not a validated intrusion detection system.

## Limitations

- The dataset does not provide explicit malicious/benign labels.
- Training used next-token prediction.
- It has not been validated for attack classification or real-world threat detection.
- The context length is limited to 256 tokens.
- Generated outputs may contain invalid or incomplete telemetry.
- The tokenizer contains additional token IDs outside the model vocabulary; inference must avoid unsupported IDs.

## Architecture

The architecture uses:

- Rotary Positional Embeddings (RoPE)
- Grouped Query Attention (GQA)
- RMSNorm
- SwiGLU feed-forward layers
- Causal self-attention

## Repository Contents

- `model.py`: Model architecture
- `model_state.pt`: Trained model weights
- `config.json`: Model configuration
- Tokenizer files: Tokenization resources

## Author

Created as an academic cybersecurity language-model project by Nithin R.
