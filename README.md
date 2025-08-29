# MAGIC
Repository for the Modality AGnostic Image Cascade (MAGIC), a segmentation framework for U-shaped networks to remain applicable to multiple different modality inputs and overlapping segmentation targets.

Modality-agnostic segmentation handled through replicated encoder branches following our previous work MAGNET and MAG-MS.

Overlapping segmentation is handled through multi-task learning where currated groups of non-overlapping structures are individually targetted to their reference labels with parallel multi-branch decoders.

All modality-specific encoders and group-specific decoders are connected to a shared bottleneck. By replicating the encoding and decoding branches, the backbone U-Net's characteristics are conserved.

---

MAGIC has been implemented here on an patch-based [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/master) with dual-self distillation as implemented in [soumbane's DualSelfDistillation](https://github.com/soumbane/DualSelfDistillation) with clinical validation for [cardiac segmentation on MR-Linac volumes](https://github.com/NRSummerfield/nnU-Net.wSD/tree/main).

This work is available on [ArXiv](https://arxiv.org/abs/2506.10797) and has been submitted to the Green journal.

![](Diagram.pdf)
