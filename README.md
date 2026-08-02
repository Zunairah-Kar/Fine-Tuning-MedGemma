# Fine-Tuning MedGemma for Brain Cancer MRI Classification

A Colab notebook that fine-tunes Google's [MedGemma-4B](https://huggingface.co/google/medgemma-4b-it) vision-language model to classify brain cancer types from MRI scans. It uses 4-bit QLoRA for memory-efficient training and evaluates the result with accuracy, a classification report, and a confusion matrix.

> **Disclaimer:** This project is for research and educational purposes only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## Overview

The notebook fine-tunes MedGemma to distinguish between three classes:

- **A:** brain glioma
- **B:** brain meningioma
- **C:** brain tumor

It frames classification as a visual question-answering task: the model is shown an MRI image and a multiple-choice prompt, and learns to respond with the correct label. Training uses LoRA adapters on top of a 4-bit quantized base model, so the whole pipeline fits on a single GPU.

## Pipeline

1. **Setup** — install dependencies and verify a CUDA GPU is available.
2. **Authentication** — log in to Hugging Face (to access MedGemma) and Kaggle (to download the dataset).
3. **Data** — download the Brain Cancer MRI dataset, load it via `imagefolder`, and split 80/20 into train/validation.
4. **Formatting** — convert each example into a chat-style user/assistant message pair.
5. **Model** — load MedGemma-4B in 4-bit (NF4) and prepare it for k-bit training.
6. **LoRA** — attach LoRA adapters (`r=8`, `alpha=16`) to all linear layers.
7. **Training** — fine-tune for 1 epoch with `SFTTrainer` and a custom data collator that masks image/pad tokens from the loss.
8. **Evaluation** — run greedy-decode inference on the validation set, parse predictions, and report accuracy, per-class metrics, and a confusion matrix.

## Requirements

- **GPU:** An NVIDIA CUDA GPU is required. The notebook is configured for an A100 (Colab); BF16 is used when supported, otherwise FP16.
- **Accounts / tokens:**
  - A [Hugging Face token](https://huggingface.co/settings/tokens) with access to `google/medgemma-4b-it` (you must accept the model's license terms).
  - A [Kaggle account and API token](https://www.kaggle.com/docs/api) to download the dataset.

### Key libraries

`transformers`, `datasets`, `evaluate`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `scikit-learn`, `kaggle`, `huggingface_hub`, `torchao`

## Usage

1. Open `Fine_Tuning_MedGemma.ipynb` in Google Colab (or a local Jupyter environment with a CUDA GPU).
2. Select an A100 or comparable GPU runtime.
3. Run the cells in order. You'll be prompted for your Hugging Face token, Kaggle username, and Kaggle API token.
4. Training runs for 1 epoch and saves the fine-tuned adapters to `./medgemma-brain-cancer-final`.
5. The evaluation cells report accuracy, a per-class classification report, and a confusion matrix on the validation set.

## Dataset

[Brain Cancer MRI Dataset](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset) by orvile, downloaded from Kaggle. Please review and comply with the dataset's license and terms of use.

## Training configuration

| Setting | Value |
| --- | --- |
| Base model | `google/medgemma-4b-it` |
| Quantization | 4-bit NF4 (double quant) |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | all linear layers |
| Epochs | 1 |
| Effective batch size | 8 (batch 1 × grad accum 8) |
| Learning rate | 2e-4, linear schedule, 3% warmup |
| Precision | BF16 (or FP16 fallback) |

## Model & data licenses

- **MedGemma** is subject to Google's [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms). You must accept them on Hugging Face before use.
- The dataset is subject to its own Kaggle license.

Ensure your use complies with both.
