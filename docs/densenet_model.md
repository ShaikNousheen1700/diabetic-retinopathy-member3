# DenseNet121 — Model and Preprocessing Explanation

## 1. What DenseNet is

DenseNet (Densely Connected Convolutional Network, Huang et al., CVPR 2017) is a CNN in which
**every layer inside a dense block receives the feature maps of all earlier layers in that block**,
joined together (concatenated).

```
Plain CNN : x0 → L1 → L2 → L3
ResNet    : output = x + F(x)            (features are ADDED)
DenseNet  : L3 input = [x0, x1, x2]      (features are CONCATENATED)
```

- **Growth rate (k = 32):** each layer adds only 32 new feature maps and reuses all earlier ones.
- **Transition layers** (1×1 conv + 2×2 average pooling) between the 4 dense blocks shrink the
  feature maps and keep the model compact.

Benefits: feature reuse, fewer parameters, strong gradient flow (short paths to every layer).

## 2. Why DenseNet suits retinal images

- DR lesions range from tiny (microaneurysms, a few pixels) to large (neovascularisation).
  Feature reuse keeps fine early-layer details available to the final classifier.
- APTOS has only 3,662 labelled images; a compact model overfits less.
- DenseNet121 is widely used in medical imaging (e.g. CheXNet for chest X-rays).

## 3. Variant: DenseNet121

| Variant | Parameters | Choice |
|---|---|---|
| **DenseNet121** | ~8.0 M (7.04 M without ImageNet top) | ✅ smallest; suits a small dataset and a free Colab GPU |
| DenseNet169 | ~14.3 M | larger, higher overfitting risk |
| DenseNet201 | ~20.2 M | not worth it for 3,662 images |

## 4. Transfer learning

ImageNet-pretrained weights are used. Training happens in two phases (`src/train.py`):

| Phase | What trains | LR | Epochs |
|---|---|---|---|
| 1 | Only the new classifier head (base frozen) | 1e-3 | 10 |
| 2 | + last dense block (`conv5_*`), BatchNorm kept frozen | 1e-5 | up to 20, early stopping (patience 5) |

Phase 1 first trains the randomly initialised head so that its large early gradients don't damage
the pretrained weights. Phase 2 then adapts the most task-specific DenseNet layers to retinas.

## 5. Model architecture (`src/model.py`)

```
Input (224, 224, 3)    RGB, values 0–255
Rescaling(1/255)       → 0–1
Normalization          → (x − ImageNet mean) / ImageNet std
DenseNet121 base       → (7, 7, 1024)
GlobalAveragePooling2D → (1024,)
Dropout(0.3)           → active only during training
Dense(5, softmax)      → (5,) class probabilities, sum = 1
```

- **Loss:** `sparse_categorical_crossentropy` (integer labels 0–4).
- **Class imbalance:** "balanced" class weights from the training split.
- The normalisation is **inside the model**, identical to `keras.applications.densenet.preprocess_input`
  (verified by `tests/test_model.py`). The saved `.keras` file therefore only needs 0–255 RGB input,
  which removes one source of training/inference mismatch in the backend.

## 6. Preprocessing (`src/preprocessing.py`)

The **same function** (`preprocess_image`) is used for training, evaluation, sample prediction and the backend.

| # | Step | Why |
|---|---|---|
| 1 | Read with OpenCV, BGR → RGB | OpenCV loads BGR; pretrained weights expect RGB |
| 2 | Crop black border (gray > 7) | removes empty pixels; images come from different cameras |
| 3 | Resize to 224×224 (INTER_AREA) | model input size; INTER_AREA is best for shrinking |
| 4 | Ben Graham (optional, `USE_BEN_GRAHAM` in `src/config.py`) | evens out lighting; team decision pending — off by default |
| 5 | ImageNet normalisation | done inside the model |

**Augmentation (training split only, `src/data.py`):** random horizontal/vertical flip, rotation ±20°,
zoom ±10%, brightness ±10%.

## 7. Data split

Stratified 70 / 15 / 15 with seed 42 → **2564 train / 549 validation / 549 test** images.
Split files are stored in `MyDrive/DR_project/splits/`; if the team provides shared split files there,
they are used instead.

## 8. Evaluation metrics (`src/evaluate.py`)

Accuracy, macro F1, weighted F1, **Quadratic Weighted Kappa** (official APTOS metric — penalises
predictions far from the true grade more), per-class precision/recall/F1, confusion matrix,
training curves, parameter count, model file size, inference time per image.
