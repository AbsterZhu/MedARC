# MedARC

Official implementation of **MedARC: Training-Free Adaptive Redundancy
Compression of Visual Tokens for 3D Medical Vision-Language Models**.

[[Paper](https://arxiv.org/abs/2607.26554)]
[[PDF](https://arxiv.org/pdf/2607.26554)]
[[Code](https://github.com/AbsterZhu/MedARC)]
[[Checkpoints](https://drive.google.com/drive/folders/REPLACE_WITH_YOUR_FOLDER_ID)]

MedARC is a training-free visual-token compression method for volumetric
medical VLMs. It combines three complementary importance signals:

- visual self-attention from the target VLM;
- visual-text similarity for query-aware relevance;
- structural saliency from the frozen 3DINO foundation model.

Important tokens are retained, while redundant tokens are merged into
contextual anchors before LLM decoding. This repository focuses on reproducing
the paper's CT-RATE report/VQA and MR-RATE report-generation evaluation.

## Setup

The reference environment uses Python 3.11, PyTorch 2.8, CUDA 12.8, and
Qwen3-VL-4B-Instruct.

```bash
git clone https://github.com/AbsterZhu/MedARC.git
cd MedARC

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Trained checkpoints

Download the CT-RATE and MR-RATE LoRA checkpoints from:

> **Google Drive:**
> [download trained weights](https://drive.google.com/drive/folders/REPLACE_WITH_YOUR_FOLDER_ID)
> — replace this placeholder with the public folder link before release.

The evaluation commands below use explicit paths, so the extracted directory
layout is flexible. Set `MODEL_PATH` to the LoRA checkpoint and `MODEL_BASE` to
its corresponding base model/checkpoint.

### 3DINO

MedARC uses frozen 3DINO features for structural saliency. Obtain the official
source and pretrained weight separately:

```bash
mkdir -p third_party
git clone https://github.com/AICONSlab/3DINO.git third_party/3DINO
```

Accept the terms at [AICONSlab/3DINO-ViT](https://huggingface.co/AICONSlab/3DINO-ViT)
and place `3dino_vit_weights.pth` at:

```text
third_party/3DINO/3dino_vit_weights.pth
```

3DINO is licensed separately under CC BY-NC-ND 4.0 and is restricted to
non-commercial research use.

## Data

Prepare the official CT-RATE and MR-RATE datasets and use JSON manifests with
paths relative to `IMAGE_FOLDER`. See [dataset/README.md](dataset/README.md) for
the expected schema.

The paper evaluates volumes resampled to `128 × 384 × 384` (DHW):

- CT-RATE: report generation and VQA;
- MR-RATE: report generation.

## Evaluate MedARC

The commands use the full MedARC model: visual attention + text similarity +
3DINO saliency. They can be launched from any directory.

### CT-RATE report generation and VQA

```bash
MODEL_PATH=/path/to/ct-rate-lora-checkpoint \
MODEL_BASE=/path/to/ct-rate-base-model-or-checkpoint \
REPORT_DATA_PATH=/path/to/validation_reports.json \
VQA_DATA_PATH=/path/to/validation_vqa.json \
IMAGE_FOLDER=/path/to/ct-rate-volumes \
GPU=0 \
bash Qwen-VL-Series-Finetune/scripts_med/eval_stage_3_MedARC_V3.sh
```

Set `EVAL_TARGET=reports` or `EVAL_TARGET=vqa` to evaluate only one task.

### MR-RATE report generation

```bash
MODEL_PATH=/path/to/mr-rate-lora-checkpoint \
MODEL_BASE=Qwen/Qwen3-VL-4B-Instruct \
REPORT_DATA_PATH=/path/to/mr_rate_test.json \
IMAGE_FOLDER=/path/to/mr-rate-volumes \
EVAL_TARGET=reports \
GPU=0 \
bash Qwen-VL-Series-Finetune/scripts_med/eval_stage_3_MedARC_V3.sh
```

### Paper configuration

The default V3 evaluator now matches the paper:

| Parameter | Value |
| --- | ---: |
| visual-attention weight `α` | `0.4` |
| visual-text weight `β` | `0.3` |
| 3DINO weight `γ` | `0.3` |
| contextual-token ratio `ρ` | `0.05` |
| cumulative-importance threshold `τ` | `0.9` |

Override the compression level without editing source code:

```bash
TAU=0.6 bash Qwen-VL-Series-Finetune/scripts_med/eval_stage_3_MedARC_V3.sh
```

The paper varies `τ` from `0.6` to `0.9`; lower values produce more aggressive
compression. Other relevant overrides are `TEXT_WEIGHT`, `VFM_WEIGHT`,
`CONTEXTUAL_RATIO`, `TEXT_TEMPERATURE`, and `VFM_TEMPERATURE`.

Use `MODE=dry-run LIMIT=1` to validate a manifest without loading the VLM.
Predictions, metrics, samples, token-retention statistics, and timing results
are written below `OUTPUT_ROOT`.

## Citation

```bibtex
@article{zhu2026medarc,
  title={MedARC: Training-Free Adaptive Redundancy Compression of Visual Tokens for 3D Medical Vision-Language Models},
  author={Zhu, Yitao and Liu, Mengjun and Fu, Yingji and Pang, Haowen and Qiu, Anqi},
  journal={arXiv preprint arXiv:2607.26554},
  year={2026}
}
```

## License

See [LICENSE.md](LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
This research code is not a medical device and must not be used for clinical
diagnosis, treatment, or patient-care decisions.
