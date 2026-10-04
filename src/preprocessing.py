"""Image preprocessing shared by training, evaluation, sample prediction and the backend.

Every image that reaches the model goes through `preprocess_image`, so training-time
and inference-time preprocessing can never drift apart.

Pipeline:
    file / bytes --(load/decode)--> RGB uint8 (H, W, 3)
                 --crop_black_border--> fundus only
                 --(optional) ben_graham--> lighting-normalised
                 --resize--> RGB uint8 (224, 224, 3)

Pixel normalisation (ImageNet mean/std) is NOT done here: it is built into the model
(see src/model.py), so the model always receives 0-255 RGB values.
"""

import cv2
import numpy as np

from src import config


def load_image(path):
    """Read an image file from disk.

    Input:  path to a PNG/JPEG file.
    Output: RGB uint8 array of shape (H, W, 3).
    Raises ValueError if the file is missing or not a readable image.
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # OpenCV loads BGR; the model expects RGB


def decode_image_bytes(data):
    """Decode raw file bytes (e.g. an HTTP upload) into an image.

    Input:  bytes of a PNG/JPEG file.
    Output: RGB uint8 array of shape (H, W, 3).
    Raises ValueError if the bytes are empty or not a valid image (corrupted file).
    """
    if not data:
        raise ValueError("Empty image data")
    buffer = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image (corrupted or unsupported format)")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def crop_black_border(img, threshold=7):
    """Remove the dark border around the round fundus.

    A pixel counts as "fundus" if its grayscale value is above `threshold`.
    We keep the smallest rectangle containing all such pixels.
    If the whole image is dark, it is returned unchanged.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def ben_graham(img, sigma=10):
    """Ben Graham's preprocessing: subtract a blurred copy to even out lighting.

    result = 4 * image - 4 * blurred(image) + 128
    Large, smooth lighting changes are removed; small details (vessels, lesions) stand out.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blurred, -4, 128)


def preprocess_image(img, img_size=config.IMG_SIZE, use_ben_graham=config.USE_BEN_GRAHAM):
    """Full preprocessing for one image.

    Input:  RGB uint8 array (H, W, 3) of any size.
    Output: RGB uint8 array (img_size, img_size, 3), ready for the model.
    """
    img = crop_black_border(img)
    if use_ben_graham:
        img = ben_graham(img)
    # INTER_AREA is the recommended interpolation when shrinking images
    return cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
