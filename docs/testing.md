# Testing Documentation

Run all tests (venv active, from the project root):

```bash
python -m pytest tests/ -v
```

**Result (2026-10-04): 38 passed** in ~90 s on CPU.

## 1. DenseNet tests (13)

| File | Test | Verifies |
|---|---|---|
| `test_preprocessing.py` | crop removes black border | fundus is cropped, roughly square |
| | all-black image unchanged | no crash on empty images |
| | output shape/type | always (224, 224, 3) uint8 |
| | Ben Graham keeps shape | optional step works |
| | load_image returns RGB | BGR→RGB conversion correct (lossless PNG round trip) |
| | missing file raises | clear `ValueError` |
| | decode valid / corrupted / empty bytes | upload decoding and its errors |
| `test_model.py` | outputs are probabilities | shape (N, 5), rows sum to 1 |
| | normalisation = `densenet.preprocess_input` | in-model normalisation identical to Keras' official one |
| | phase 1 / phase 2 trainable layers | only head trains first; last block (no BatchNorm) in phase 2 |
| `test_pipeline.py` | stratified, disjoint splits | 70/15/15, no image in two splits, class ratios kept |
| | class weights | rarer class → larger weight |
| | full pipeline | 50 synthetic images: split → train both phases → save → reload → evaluate → predict (path and bytes give same result) |

## 2. Backend tests (25) — `test_api.py`

Most use a fake predictor (fast; can simulate crashes). The last 3 use the real trained model.

| Required case | Test(s) | Expected |
|---|---|---|
| Valid fundus image | valid PNG, valid JPEG (`.JPG` upper-case) | 200, `success: true` |
| Response structure | exact top-level and `prediction` keys | fixed JSON shape |
| Missing file | no body; wrong field name; empty file | 400 `MISSING_FILE` / `EMPTY_FILE` |
| Unsupported file | `.txt`; GIF renamed to `.png` | 415 `UNSUPPORTED_FILE_TYPE` |
| Corrupted image | text renamed to `.png`; truncated PNG; truncated JPEG | 422 `CORRUPTED_IMAGE` |
| Oversized image | 11 MB file; PNG header claiming 10000×10000 | 413 `FILE_TOO_LARGE` / `IMAGE_TOO_LARGE` |
| Too small | 40×40 image | 422 `IMAGE_TOO_SMALL` |
| Model error | predictor raises | 500 `PREDICTION_FAILED`, `/health` still 200 |
| Model missing | nonexistent model path | `/health` degraded, `/predict` 503 `MODEL_NOT_LOADED` |
| Demo page | `GET /demo` | 200 HTML using field name `file` |
| Wrong route/method | `GET /predict`, `GET /nope` | 405 / 404 in the same JSON error format |
| Repeated requests | 20 sequential; 16 concurrent (8 threads); error then success | all succeed, identical results |
| Real model | `/model-info`; API result == direct `DRPredictor` for 5 real fundus images; 8 concurrent requests | identical predictions |

### Bug found by the tests
A truncated JPEG made Pillow raise `OSError`, which was not caught → the API returned 500 instead
of 422. Fixed in `backend/validation.py`.

## 3. Manual test with the running server (curl)

| Request | Result |
|---|---|
| `GET /health` | `{"status":"ok","model_loaded":true}` |
| `POST /predict` real fundus (2588×1958 PNG) | 200, prediction + probabilities |
| no file | 400 `MISSING_FILE` |
| `requirements.txt` | 415 `UNSUPPORTED_FILE_TYPE` |
| text file renamed `.png` | 422 `CORRUPTED_IMAGE` |
| 11 MB file | 413 `FILE_TOO_LARGE` |
| 10 repeated requests | all 200, ~0.3 s each |

First request was 2.9 s before adding a warm-up prediction at startup; after it, the first
request is ~0.3 s like the rest.

## 4. Real-model sample predictions (CPU, local)

| Image | True | Predicted | Confidence |
|---|---|---|---|
| class0_31360e44ac64 | No DR | No DR ✅ | 0.954 |
| class1_4aa07d720638 | Mild | Mild ✅ | 0.609 |
| class2_a688f20f8895 | Moderate | Mild ❌ | 0.834 |
| class3_42cc993f23a9 | Severe | No DR ❌ | 0.323 |
| class4_cd54d022e37d | Proliferative DR | Moderate ❌ | 0.531 |

Identical to the predictions made on Colab, confirming the local model and preprocessing match training.
These 5 images are examples only; accuracy is measured on the 549-image test split (see README).
