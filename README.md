# Aimonk Multilabel Classification

## 1. Framework Used
The solution is implemented using **PyTorch** and torchvision.

---

## 2. Model Architecture
- Backbone: **ResNet18**
- Pretrained on **ImageNet**
- Final fully connected layer replaced to match the number of attributes.
- The model is fine-tuned on the given dataset.

The architecture is not built from scratch, as required.

---

## 3. Fine-Tuning Strategy
- ImageNet pretrained weights are loaded.
- Final classification layer is replaced.
- Entire network is fine-tuned using a low learning rate (1e-4).
- Optimizer: Adam
- Loss: BCEWithLogitsLoss

The model is NOT trained from scratch.

---

## 4. Handling Missing Labels (NA)

Images containing "NA" values are NOT ignored.

Instead:
- A binary mask is created for each sample.
- Loss is computed only for available attributes.
- Missing attributes do not contribute to the loss.

This ensures:
- Images are preserved.
- Training remains unbiased.

---

## 5. Handling Class Imbalance

Since the dataset is skewed:
- Positive class weights are computed for each attribute.
- These weights are passed to BCEWithLogitsLoss using the `pos_weight` parameter.
- Rare attributes receive higher penalty when misclassified.

This reduces bias toward dominant classes.

---

## 6. Data Preprocessing

- Resize to 224x224
- Random Horizontal Flip
- Normalization using ImageNet statistics

---

## 7. Training Details

- Batch size: 16
- Epochs: 10
- Optimizer: Adam
- Learning rate: 1e-4
- Device: GPU (Tesla T4)

---

## 8. Outputs

- multilabel_model.pth (trained weights)
- loss_plot.png (training loss curve)
- train.py (training code)
- infer.py (inference code)

Loss curve formatting:
- X-axis: iteration_number
- Y-axis: training_loss
- Title: Aimonk_multilabel_problem

---

## 9. Possible Improvements (Not Implemented Due to Time Constraint)

- Validation split and evaluation metrics (mAP, F1-score)
- Learning rate scheduler
- Focal Loss for severe imbalance
- Weighted Random Sampler
- Stronger augmentations (Color Jitter, Random Crop)
- Early stopping
- Threshold tuning per attribute

---

## 10. Cleanliness and Modularity

The project is structured into:
- Dataset class (data loading + NA masking)
- Training script
- Inference script
- Loss plotting
- Model saving

The code is modular, readable, and production-aligned.

---

## Conclusion

This implementation:
- Uses a pretrained architecture
- Handles missing labels correctly
- Handles class imbalance
- Produces required outputs
- Maintains clean modular structure

The model successfully converges as shown in the loss curve.

---
