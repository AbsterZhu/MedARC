# Third-party notices

This repository combines project-specific research code with third-party
software and model interfaces. This file is an attribution summary, not a
replacement for the applicable license texts.

## Qwen-VL-Series-Finetune

`Qwen-VL-Series-Finetune/` is derived from
[2U1/Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune),
which is licensed under Apache License 2.0. The full license is retained at
`Qwen-VL-Series-Finetune/LICENSE`.

Qwen model weights are separate downloads and remain subject to the license of
the selected model repository.

## 3DINO and 3DINO-ViT

3DINO code and 3DINO-ViT weights are provided by
[AICONSlab/3DINO](https://github.com/AICONSlab/3DINO) under CC BY-NC-ND 4.0.
The upstream terms restrict use to non-commercial research and prohibit
distribution of modified copies. Consequently, the official source and weights
are not vendored in the publishable MedARC tree; users obtain them directly and
accept the upstream terms.

If 3DINO is used, cite:

```bibtex
@article{xu3dino2025,
  title={A generalizable 3D framework and model for self-supervised learning in medical imaging},
  author={Xu, Tony and Hosseini, Sepehr and Anderson, Chris and Rinaldi, Anthony and Krishnan, Rahul G. and Martel, Anne L. and Goubran, Maged},
  journal={npj Digital Medicine},
  year={2025},
  doi={10.1038/s41746-025-02035-w}
}
```

## Datasets

CT-RATE and MR-RATE data and derived manifests are not licensed by this source
repository. Users are responsible for obtaining authorized access, preserving
de-identification, and following the official dataset terms. Never commit raw
medical volumes, patient identifiers, credentials, or private download links.
