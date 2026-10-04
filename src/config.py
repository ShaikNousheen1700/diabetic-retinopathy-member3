"""Shared settings for the DenseNet121 model.

Values follow docs/common_experiment_settings.md so that DenseNet, ResNet and
EfficientNet are trained and evaluated under the same conditions.
"""

# --- Dataset ---
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# --- Image / preprocessing ---
IMG_SIZE = 224              # DenseNet121 was pretrained on 224 x 224 ImageNet images
USE_BEN_GRAHAM = False      # team decision pending (settings doc, section 4)

# ImageNet statistics used by keras.applications.densenet.preprocess_input ("torch" mode).
# The model applies them itself, so callers pass plain 0-255 RGB images.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Split ---
SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# --- Model ---
DROPOUT = 0.3
FINE_TUNE_FROM = "conv5_block1_0_bn"   # first layer of DenseNet121's last dense block

# --- Training (phase 1 = head only, phase 2 = fine-tuning) ---
BATCH_SIZE = 32
PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 20
PHASE2_LR = 1e-5
EARLY_STOP_PATIENCE = 5

# --- Output file names ---
MODEL_NAME = "densenet121"
MODEL_FILENAME = f"{MODEL_NAME}_dr.keras"
MODEL_INFO_FILENAME = f"{MODEL_NAME}_dr_info.json"
