
# GHR 2.0 Hackathon Report  
## Offroad Semantic Scene Segmentation  

**Team Name:** Black Bone  
**Project Name:** DesertScene Robust Segmentation  
**Track:** Segmentation  

---

# 1. Project Summary

This project focuses on training a semantic segmentation model on a synthetic desert dataset and evaluating its ability to generalize to unseen desert environments.

**Primary Metric:** Mean Intersection over Union (mIoU)  
**Best Validation mIoU Achieved:** **0.2632 (Epoch 14)**  

### Key Challenges
- Severe class imbalance  
- Small-object segmentation difficulty  
- Texture similarity (Rocks vs Landscape)  
- Overfitting during extended training  

---

# 2. Methodology

## Model Architecture

- **Backbone:** Vision Transformer (DINOv2 encoder)  
- **Decoder:** Segmentation upsampling head  
- **Loss Function:** Class-weighted Cross Entropy  
- **Optimizer:** AdamW  
- **Learning Rate:** 0.001 with cosine decay  
- **Mixed Precision:** Enabled (AMP)  
- **Checkpoint Strategy:** Best validation mIoU selected  
<img width="1465" height="712" alt="image" src="https://github.com/user-attachments/assets/859c834d-a859-4267-bbbd-16779848e100" />
<img width="1474" height="693" alt="image" src="https://github.com/user-attachments/assets/14aac49b-d7db-43f3-ab2f-2735a327265b" />

---

# 3. Training Curve Analysis

## 3.1 Training vs Validation Loss
<img width="2100" height="1500" alt="metrics" src="https://github.com/user-attachments/assets/3f28b6b7-0b63-4aaf-8c6a-2abe9c277f49" />



Validation loss decreases until approximately **Epoch 12**, then increases significantly — indicating overfitting.

---

## 3.2 Training vs Validation mIoU

<img width="2100" height="900" alt="per_class_iou" src="https://github.com/user-attachments/assets/114c0d6b-bc45-42d3-9152-60c16a6391e2" />


Validation mIoU peaks at:

> **Epoch 14 → 0.2632**

After this point, performance declines due to overfitting.  
The best checkpoint was selected at this peak.

---

## 3.3 Dice & Accuracy Trends


Both Dice and Accuracy follow similar behavior, peaking between **Epoch 10–14** and declining afterward.

---

# 4. Per-Class IoU Analysis


### Best Epoch Per-Class IoU Summary

| Class | IoU |
|--------|------|
| Sky | ~0.69 |
| Landscape | ~0.48 |
| Trees | ~0.28 |
| Dry Grass | ~0.27 |
| Rocks | ~0.23 |
| Ground Clutter | ~0.15 |
| Logs | ~0.13 |
| Lush Bushes | ~0.08 |
| Dry Bushes | ~0.02 |

### Observations

- Large-area classes perform strongly.
- Minority vegetation classes remain unstable.
- Dry Bushes collapse due to extreme underrepresentation.
- Logs struggle due to occlusion and small size.
- Clear impact of class imbalance.

---

# 5. Failure Case Analysis

## Case 1: Dry Bushes Misclassified as Landscape

Reason: Visual similarity + class imbalance.


---

## Case 2: Logs Under-Segmented

Reason: Small object size and occlusion

---

## Case 3: Rocks vs Landscape Confusion

Reason: Texture similarity and boundary blending.


---

# 6. Training Log Verification


The log confirms automatic saving of the best model at:

> Validation mIoU = **0.2632**

---

# 7. Generalization to Unseen Biome

The model was evaluated on unseen desert images.

### Observations

- Strong Sky and Landscape segmentation maintained.
- Moderate degradation in minority vegetation classes.
- Structural terrain understanding preserved.

*(Insert unseen biome prediction image if available)*

---

# 8. Challenges & Solutions

### Overfitting
Detected via divergence between training and validation curves.  
Solved by selecting early checkpoint.

### Class Imbalance
Addressed using class-weighted loss.

### Minority Class Instability
Monitored per-class IoU trends.

---

# 9. Conclusion

This project demonstrates:

- Successful transformer-based segmentation training  
- Peak validation mIoU of **0.2632**  
- Clear identification of overfitting behavior  
- Detailed per-class performance analysis  

Primary limitation remains instability in minority vegetation classes.
<img width="548" height="850" alt="Screenshot 2026-02-19 013134" src="https://github.com/user-attachments/assets/2d72a379-d5ad-4694-b181-ff867a7d96b8" />

---

# 10. Future Work

- Focal loss for hard pixel mining  
- Stronger data augmentation (CutMix, Copy-Paste)  
- Oversampling minority classes  
- Multi-scale training  
- Domain adaptation techniques  

---

**End of Report**
