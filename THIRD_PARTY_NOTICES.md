# Third-party notices

SpikeBirkhoff is maintained by Zhiqi Cai. The root [MIT license](LICENSE),
Copyright (c) 2026 Zhiqi Cai, covers the original additions in this project.
Adapted third-party portions retain the copyright notices and license terms
listed below. The root license does not replace those terms.

## Adapted source

The spiking backbone (`model.py` and its layers), training engine
(`training_engine.py`), and dataset utilities build on the following projects.
Source-file notices identify the applicable origins and modifications.

| Upstream project | Scope of reuse | License and attribution |
| --- | --- | --- |
| [STAtten](https://github.com/Intelligent-Computing-Lab-Panda/STAtten) | Spiking backbone, layers, training and dataset scaffolding | [MIT](LICENSES/STAtten-MIT.txt); Copyright (c) 2025 Donghyun Lee |
| [Spike-Driven-Transformer](https://github.com/BICLab/Spike-Driven-Transformer) | Backbone and training/dataset components inherited through STAtten | [Apache-2.0](LICENSES/Spike-Driven-Transformer-Apache-2.0.txt); Man Yao, Jiakui Hu, Zhaokun Zhou, Li Yuan, Yonghong Tian, Bo Xu, and Guoqi Li |
| [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) | Training and data utilities adapted through the upstream code | [Apache-2.0](LICENSES/timm-Apache-2.0.txt); Copyright 2019 Ross Wightman |

STAtten was checked at commit
`e9048a18b02cdaf824ac2ad82f2e6f5bfcf50134`. Its former
`Intelligent-Computing-Lab-Yale/STAtten` address redirects to the repository
above. Its README acknowledges Spike-Driven-Transformer and SpikingJelly.
The included timm license is from version `0.6.12`; the
Spike-Driven-Transformer license was retrieved from its official `main`
branch on September 15, 2026.

Keep the included license texts and relevant source notices when
redistributing adapted portions. Apache-2.0 also requires prominent notices
on modified files and preservation of applicable upstream attribution and
NOTICE content.

## External dependencies

[SpikingJelly](https://github.com/fangwei123456/spikingjelly) is installed as
an external dependency. Its source is not bundled in this repository. The
[Open-Intelligence Open Source License v1.0](LICENSES/SpikingJelly-OI-1.0.txt)
is included for reference from version `0.0.0.0.12`. That license governs
SpikingJelly independently: sections II, III, and VI address retained notices
on redistribution, and section V addresses disclosure for commercial use.
The upstream [license guide](https://github.com/fangwei123456/spikingjelly/blob/master/LICENSES/README.md)
provides further context.

timm and other separately installed packages retain their own licenses.
Licenses for source code do not grant rights to third-party datasets or
pretrained weights.

## Research attribution

Please acknowledge the upstream research when using the associated methods:

- Donghyun Lee, Yuhang Li, Youngeun Kim, Shiting Xiao, and Priyadarshini Panda.
  [Spiking Transformer with Spatial-Temporal Attention](https://arxiv.org/abs/2409.19764).
  CVPR 2025, pp. 13948–13958.
- Man Yao, Jiakui Hu, Zhaokun Zhou, Li Yuan, Yonghong Tian, Bo Xu, and Guoqi Li.
  [Spike-driven Transformer](https://openreview.net/forum?id=9FmolyOHi5).
  NeurIPS 2023.
- Ross Wightman.
  [PyTorch Image Models](https://doi.org/10.5281/zenodo.4414861), 2019.
- Wei Fang and collaborators.
  [SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence](https://doi.org/10.1126/sciadv.adi1480).
  Science Advances 9(40), eadi1480, 2023.

Academic citations supplement the license notices; they do not replace them.
