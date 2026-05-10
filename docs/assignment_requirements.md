# HW2 Requirements Summary

## Task 1: Flowers102 Classification

- Dataset: 102 Category Flower Dataset.
- Baseline: modify ResNet-18 or ResNet-34 output layer and initialize from ImageNet pretrained weights.
- Fine-tuning: train the new classifier head from scratch and use a smaller learning rate for the rest of the network.
- Hyperparameter analysis: compare training steps/epochs, learning rates, and combinations.
- Pretraining ablation: compare ImageNet pretrained initialization with random initialization.
- Attention model: add an attention module such as SE-block or CBAM, or train a light ViT/Swin model; compare Accuracy with baseline.

## Task 2: Road Vehicle Detection and Tracking

- Dataset: Road Vehicle Images Dataset.
- Detector: fine-tune YOLOv8 or a comparable modern one-stage detector.
- Video: prepare a 10-30 second test video.
- Tracking: run frame-by-frame inference with multi-object tracking, output bounding boxes, classes, and stable Tracking IDs.
- Occlusion analysis: choose a clip with occlusion or dense crossing, visualize 3-4 consecutive frames, and discuss whether IDs remain stable or switch.
- Line counting: define a virtual line and count objects crossing it using box centers and Tracking ID continuity.

## Task 3: U-Net Segmentation From Scratch

- Dataset: Stanford Background Dataset.
- Model: implement U-Net manually with encoder, decoder, and skip connections.
- No pretrained weights.
- Loss engineering: implement Dice Loss manually.
- Compare validation mIoU under three losses:
  - Cross-Entropy Loss only.
  - Dice Loss only.
  - Cross-Entropy Loss + Dice Loss.

## Submission

- Submit PDF report only.
- Report must include model structures, datasets, experimental settings, results, and visualizations.
- Include wandb or swanlab screenshots/curves for train/validation loss and validation Accuracy or mAP.
- Put code in a public GitHub repository with clear README.
- Upload trained weights to cloud storage and include repository and weight links in the report.
- Report cover page should list names, student IDs, and division of work.
