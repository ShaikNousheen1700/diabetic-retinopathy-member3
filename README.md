# Diabetic Retinopathy Severity Classification — DenseNet + Backend (Member 3)

Part of the team project **"Comparative Analysis of Deep Learning Models for Diabetic Retinopathy Severity Classification"** (DenseNet vs ResNet vs EfficientNet).

- **Dataset:** APTOS 2019 Blindness Detection (5 classes: 0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative DR)
- **Framework:** TensorFlow / Keras (model saved as `.keras`)
- **Training:** Google Colab (GPU)
- **Backend:** prediction API (`/health`, `/model-info`, `/predict`)

## Project structure

| Folder | Contents |
|---|---|
| `src/` | Preprocessing, model and inference code |
| `notebooks/` | Colab training / evaluation notebooks |
| `backend/` | Prediction API |
| `tests/` | Backend and inference tests |
| `models/` | Trained model files (not committed) |
| `data/` | Dataset (not committed — downloaded on Colab) |
| `results/` | Metrics, plots, confusion matrices |
| `docs/` | API docs, architecture, demo notes |

*Work in progress — sections will be added as each step is built and tested.*
