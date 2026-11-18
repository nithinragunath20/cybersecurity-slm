# Cybersecurity Small Language Model (SLM)

An approximately 80.66-million-parameter custom Transformer language model trained from scratch on structured cybersecurity telemetry based on DARPA Transparent Computing Common Data Model v18 (CDM18).

## Project Links

- **GitHub Source Code:** https://github.com/nithinragunath20/cybersecurity-slm
- **Hugging Face Trained Model:** https://huggingface.co/nithinragunath/cybersecurity-slm
- **Hugging Face Dataset:** https://huggingface.co/datasets/nithinragunath/DT

## 1. Project Overview

This project explores the use of a small language model for learning patterns in structured cybersecurity telemetry.

The model is trained to predict the next token in telemetry records. It learns the structure and patterns of the supplied CDM18 data and can generate continuations of similar structured records.

The model architecture was implemented from scratch using PyTorch.

**Important:** This is an experimental telemetry language-modelling project. It is not a validated intrusion detection system.

## 2. Dataset

**Dataset:** DARPA Transparent Computing (TC) CDM18 telemetry.

The dataset contains 1,000,000 structured records.

| Dataset Split | Number of Records |
|---|---:|
| Training | 700,000 |
| Validation | 150,000 |
| Testing | 150,000 |
| **Total** | **1,000,000** |

Dataset fields include:
- Record type
- CDM version
- Source
- JSON event details

Dataset link: https://huggingface.co/datasets/nithinragunath/DT

The dataset does not contain explicit malicious or benign labels.

## 3. Model Architecture

The project uses a custom decoder-only Transformer architecture inspired by modern Qwen3 design concepts.

It does not use pretrained Qwen model weights.

| Parameter | Configuration |
|---|---:|
| Total parameters | 80,658,432 |
| Transformer layers | 4 |
| Hidden dimension | 256 |
| Attention heads | 8 |
| KV groups | 2 |
| Feed-forward dimension | 768 |
| Maximum context length | 256 tokens |
| Vocabulary size | 151,643 |
| Tokenizer | Qwen3 tokenizer |

### Architecture Components

- Rotary Positional Embeddings (RoPE)
- Grouped Query Attention (GQA)
- RMSNorm
- SwiGLU Feed-Forward Network
- Causal Self-Attention

## 4. Training Configuration

| Setting | Value |
|---|---:|
| Framework | PyTorch |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Sequence length | 256 |
| Selected checkpoint | Iteration 76,000 |

The model was trained from randomly initialized weights using next-token prediction.

The best checkpoint was selected based on validation loss.

## 5. Evaluation Results

| Evaluation Metric | Result |
|---|---:|
| Training loss | 0.5774 |
| Validation loss | 0.5374 |
| Sampled test loss | 0.6253 |

The sampled test loss was calculated using 50 randomly selected test batches.

These values represent language-modelling loss. They do not represent attack detection accuracy, precision, recall, or F1-score.

## 6. Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NumPy
- Jupyter Notebook
- NVIDIA CUDA

## 7. Repository Structure

```text
cybersecurity-slm/
│
├── README.md
├── cybersecurity_slm_training.ipynb
├── model.py
├── config.json
├── tokenizer.json
└── tokenizer_config.json
```

The trained model weights are hosted separately on Hugging Face to avoid storing large binary files in GitHub.

## 8. How to Run

1. Clone the repository:

```bash
git clone https://github.com/nithinragunath20/cybersecurity-slm.git
```

2. Open the project in VS Code or Jupyter Notebook.

3. Install the required Python libraries used in the training notebook:

```bash
pip install torch transformers datasets numpy tqdm huggingface_hub
```

4. Open `cybersecurity_slm_training.ipynb`.

5. Follow the notebook cells to prepare the dataset, tokenize the records, build the model, and train it.

For the exported model weights and configuration, visit:

https://huggingface.co/nithinragunath/cybersecurity-slm

## 9. Limitations

- The dataset does not contain explicit malicious or benign labels.
- The model uses next-token prediction rather than supervised attack classification.
- It has not been validated for real-world intrusion detection.
- The maximum context length is 256 tokens.
- Generated telemetry may be incomplete or invalid.
- The tokenizer contains additional token IDs outside the model vocabulary; inference must prevent unsupported IDs from being generated or passed to the model.

## 10. Future Scope

- Explore larger cybersecurity telemetry datasets.
- Experiment with longer context lengths.
- Evaluate generated telemetry more extensively.
- Explore supervised fine-tuning if suitable labelled data becomes available.
- Investigate integration with cybersecurity analysis tools.

## Author

**Nithin R.**

Academic project – Cybersecurity Small Language Model (SLM).
