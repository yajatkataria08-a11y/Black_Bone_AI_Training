=====================================================================
GHR 2.0 Hackathon Report
Offroad Semantic Scene Segmentation
=====================================================================

Team Name: [Your Team Name]
Project Name: DesertScene Robust Segmentation
Track: Segmentation
Platform: Falcon Synthetic Desert Dataset

=====================================================================
1. TITLE & PROJECT SUMMARY
=====================================================================

This project focuses on training a semantic segmentation model
on a synthetic desert dataset and evaluating its ability to
generalize to unseen desert environments.

The goal is to achieve strong pixel-level classification
performance while maintaining robustness across biome shifts.

Primary Evaluation Metric:
Mean Intersection over Union (mIoU)

Key Challenges:
- Severe class imbalance
- Small object segmentation difficulty
- Texture similarity between terrain classes
- Overfitting during training

=====================================================================
2. METHODOLOGY
=====================================================================

2.1 Dataset

Training Images: 2857
Validation Images: 317
Test Set: Unseen desert biome

Classes:
Trees, Lush Bushes, Dry Grass, Dry Bushes,
Ground Clutter, Logs, Rocks, Landscape, Sky, Background

Observation:
Landscape and Sky dominate pixel distribution.
Vegetation subclasses and Logs are underrepresented.

-------------------------------------------------------------

2.2 Model Architecture

Backbone:
Vision Transformer (DINOv2 encoder)

Decoder:
Upsampling segmentation head

Loss Function:
Class-weighted Cross Entropy

Optimizer:
AdamW

Learning Rate:
0.001 with cosine decay scheduling

Mixed Precision:
Enabled (AMP)

Training Epochs:
50 (Best checkpoint selected early)

-------------------------------------------------------------

2.3 Training Strategy

- Class weighting to address imbalance
- Validation monitoring for early stopping
- Best model saved based on validation mIoU
- Cosine LR scheduling to stabilize convergence

=====================================================================
3. RESULTS & PERFORMANCE METRICS
=====================================================================

3.1 Training Curve Analysis

[Insert Figure 1: Training vs Validation Loss Curve]

Observation:
Training loss decreases steadily.
Validation loss decreases until ~Epoch 12,
then increases significantly.

Conclusion:
Clear overfitting after Epoch 12.

-------------------------------------------------------------

[Insert Figure 2: Training vs Validation mIoU Curve]

Validation mIoU peaks at:

Epoch 14 → 0.2632 (Best Model)

After this point, validation performance declines.

Final selected checkpoint:
Epoch 14 (Val mIoU = 0.2632)

-------------------------------------------------------------

[Insert Figure 3: Dice & Accuracy Trends]

Dice and Accuracy follow similar patterns:
Peak performance occurs between Epoch 10–14.

=====================================================================
3.2 PER-CLASS IoU ANALYSIS
=====================================================================

Best Validation mIoU: 0.2632

Per-Class IoU at Best Epoch:

Sky:            0.63 – 0.69
Landscape:      0.47 – 0.48
Trees:          0.28
Dry Grass:      0.24 – 0.27
Rocks:          0.22 – 0.23
Ground Clutter: 0.15
Logs:           0.13
Lush Bushes:    ~0.08
Dry Bushes:     ~0.02

[Insert Figure 4: Per-Class IoU Across Epochs]

Observations:

- Sky and Landscape perform strongly due to dominance.
- Minority vegetation classes remain unstable.
- Dry Bushes collapse due to extreme underrepresentation.
- Logs suffer from small object size and occlusion.

=====================================================================
4. FAILURE CASE ANALYSIS
=====================================================================

Failure Case 1: Dry Bushes Collapse
Reason:
Visual similarity to Landscape and class imbalance.

[Insert Figure 5: RGB vs Ground Truth vs Prediction – Dry Bushes]

-------------------------------------------------------------

Failure Case 2: Logs Under-Segmentation
Reason:
Small object size and occlusion by vegetation.

[Insert Figure 6: RGB vs Ground Truth vs Prediction – Logs]

-------------------------------------------------------------

Failure Case 3: Rocks vs Landscape Confusion
Reason:
Similar texture and boundary blending.

[Insert Figure 7: RGB vs Ground Truth vs Prediction – Rocks]

-------------------------------------------------------------

Root Cause Summary:
The model prioritizes dominant terrain classes.
Minority classes require stronger regularization
or data balancing techniques.

=====================================================================
5. CHALLENGES & SOLUTIONS
=====================================================================

Challenge 1: Overfitting After Epoch 12
Solution:
Selected best checkpoint using validation mIoU.
Applied learning rate scheduling.

-------------------------------------------------------------

Challenge 2: Class Imbalance
Solution:
Implemented class-weighted loss to increase penalty
for minority class misclassification.

-------------------------------------------------------------

Challenge 3: Minority Class Instability
Solution:
Monitored per-class IoU and analyzed collapse behavior.

=====================================================================
6. GENERALIZATION TO UNSEEN BIOME
=====================================================================

The model was evaluated on unseen desert test images.

Observations:
- Strong terrain and sky segmentation maintained.
- Vegetation classes degrade slightly.
- Structural layout understanding preserved.

[Insert Figure 8: Prediction on Unseen Biome Image]

Conclusion:
Model generalizes reasonably well to biome shifts
but remains sensitive to minority class variations.

=====================================================================
7. CONCLUSION
=====================================================================

This project demonstrates:

- Successful transformer-based segmentation training
- Peak validation mIoU of 0.2632
- Effective overfitting detection and control
- Detailed per-class analysis
- Robust dominant terrain understanding

The primary limitation remains minority vegetation
class segmentation due to imbalance.

=====================================================================
8. FUTURE WORK
=====================================================================

Potential improvements:

- Focal loss for hard pixel mining
- Stronger data augmentation (CutMix / Copy-Paste)
- Oversampling minority-class patches
- Multi-scale training strategy
- Domain adaptation techniques

=====================================================================
END OF REPORT
=====================================================================
