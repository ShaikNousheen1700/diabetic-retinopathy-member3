# Common Experimental Settings

**Project:** Comparative Analysis of Deep Learning Models for Diabetic Retinopathy Severity Classification
**Models:** DenseNet121 (Member 3) · ResNet (Member _) · EfficientNet (Member _)
**Status:** 🟡 PROPOSED — waiting for team agreement
**Proposed by:** Member 3 (DenseNet + Backend)

> **Why this document exists:** a comparison is only fair if every model is trained and tested
> under the same conditions. If one model gets a bigger image size, a different test set or more
> epochs, we cannot tell whether it won because it is a better architecture or because it had an
> easier setup. Every member should follow the settings below. Any change must be agreed by all
> members and recorded in the [Change log](#change-log).

---

## 1. Framework

| Setting | Value |
|---|---|
| Framework | **TensorFlow / Keras** |
| Model file format | **`.keras`** (not `.pth` — that is PyTorch) |
| Training environment | Google Colab (GPU runtime) |
| Random seed | **42** (Python `random`, NumPy, TensorFlow) |

## 2. Dataset

| Setting | Value |
|---|---|
| Dataset | APTOS 2019 Blindness Detection (Kaggle) |
| Labelled data used | `train.csv` + `train_images/` — 3,662 images |
| Classes | 0 No DR · 1 Mild · 2 Moderate · 3 Severe · 4 Proliferative DR |
| Kaggle `test_images/` | **Not used for evaluation** (it has no labels) |

## 3. Data split

| Setting | Value |
|---|---|
| Split | **70% train / 15% validation / 15% test** |
| Method | Stratified by `diagnosis` (each split keeps the same class ratio) |
| Seed | 42 |
| Split files | `train_split.csv`, `val_split.csv`, `test_split.csv` (columns: `id_code`, `diagnosis`) |
| Created by | **One person only**, then shared with everyone — *nobody re-splits on their own* |

The **test split is used only once per model**, for the final evaluation. Model choices
(epochs, early stopping, etc.) are made using the validation split only.

## 4. Preprocessing (identical for training, evaluation and the backend)

| # | Step | Value |
|---|---|---|
| 1 | Read image | OpenCV, convert **BGR → RGB** |
| 2 | Crop black border | Crop to the fundus circle (remove near-black border pixels) |
| 3 | Resize | **224 × 224** |
| 4 | Ben Graham preprocessing | ☐ Yes ☐ No — *team to decide* |
| 5 | Normalisation | Each model's own Keras `preprocess_input` (DenseNet / ResNet / EfficientNet), because each set of pretrained weights expects its own normalisation |

## 5. Data augmentation (training split only)

Random horizontal flip, random vertical flip, random rotation (±20°), random zoom (±10%),
random brightness (±10%). **No augmentation** on validation, test or backend images.

## 6. Model setup

| Setting | Value |
|---|---|
| Pretrained weights | ImageNet (transfer learning) |
| Input shape | (224, 224, 3) |
| Classifier head | GlobalAveragePooling2D → Dropout(0.3) → Dense(5, softmax) |
| Loss | `sparse_categorical_crossentropy` |
| Class imbalance | Class weights computed from the **train split** (`sklearn` "balanced") |

## 7. Training

| Setting | Phase 1 — head only | Phase 2 — fine-tuning |
|---|---|---|
| Base model | Frozen | Top block(s) unfrozen |
| Optimizer | Adam | Adam |
| Learning rate | 1e-3 | 1e-5 |
| Epochs | 10 | Up to 20 |
| Batch size | 32 | 32 |
| Early stopping | — | Monitor `val_loss`, patience 5, restore best weights |

Each member records the number of epochs actually trained and the training time.

## 8. Evaluation (on the shared test split)

Every model reports:

- Accuracy
- Macro F1-score
- **Quadratic Weighted Kappa** (official APTOS metric)
- Per-class precision, recall and F1 (`classification_report`)
- Confusion matrix (5 × 5 plot)
- Training / validation accuracy and loss curves
- Number of parameters, model file size, average inference time per image

## 9. Deliverables per member

| Item | Location / name |
|---|---|
| Trained model | `<model>_dr.keras` (e.g. `densenet121_dr.keras`) |
| Metrics | `results/<model>_metrics.json` |
| Plots | `results/<model>_confusion_matrix.png`, `results/<model>_training_curves.png` |
| Notebook | `notebooks/<model>_training.ipynb` |

---

## Decisions still needed from the team

- [ ] Who creates and shares the split files?
- [ ] Ben Graham preprocessing — yes or no?
- [ ] Confirm 224 × 224 input size
- [ ] Confirm training settings (section 7)
- [ ] Fill in member names for ResNet and EfficientNet

## Change log

| Date | Change | Agreed by |
|---|---|---|
| 2026-10-04 | Initial proposal | Member 3 (pending) |
