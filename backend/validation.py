"""Upload validation — runs BEFORE the image reaches the model.

Checks, in order (cheapest first):
    1. file present                     -> 400 MISSING_FILE
    2. file extension allowed           -> 415 UNSUPPORTED_FILE_TYPE
    3. file not empty                   -> 400 EMPTY_FILE
    4. file size within limit           -> 413 FILE_TOO_LARGE
    5. real image content (PNG / JPEG)  -> 415 UNSUPPORTED_FILE_TYPE or 422 CORRUPTED_IMAGE
    6. image dimensions sensible        -> 422 IMAGE_TOO_SMALL / 413 IMAGE_TOO_LARGE
    7. full decode with OpenCV          -> 422 CORRUPTED_IMAGE

Step 5 reads only the image header with Pillow (fast, no pixel decoding), so a tiny file
that claims to be 100,000 x 100,000 pixels is rejected before it can use gigabytes of memory.
Checking the content (not just the extension) catches e.g. a .txt file renamed to .png.
"""

import io
import os
import warnings

from PIL import Image, UnidentifiedImageError

from backend import settings
from backend.errors import APIError
from src.preprocessing import decode_image_bytes


def check_filename(filename):
    """Steps 1-2. Returns the lower-case extension."""
    if not filename:
        raise APIError(400, "MISSING_FILE", "No file uploaded. Send the image in a form field named 'file'.")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_EXTENSIONS))
        raise APIError(415, "UNSUPPORTED_FILE_TYPE", f"File type '{ext or 'none'}' is not supported. Allowed: {allowed}.")
    return ext


def check_size(data):
    """Steps 3-4."""
    if len(data) == 0:
        raise APIError(400, "EMPTY_FILE", "The uploaded file is empty.")
    if len(data) > settings.MAX_FILE_SIZE_BYTES:
        raise APIError(413, "FILE_TOO_LARGE",
                       f"File is larger than the {settings.MAX_FILE_SIZE_MB:g} MB limit.")


def check_header(data):
    """Steps 5-6 using only the image header. Returns (format, width, height)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as img:
                fmt, (width, height) = img.format, img.size
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise APIError(413, "IMAGE_TOO_LARGE", "Image resolution is too large.")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        # UnidentifiedImageError: not an image at all; OSError: truncated file (e.g. cut-off JPEG);
        # SyntaxError/ValueError: Pillow's errors for malformed headers
        raise APIError(422, "CORRUPTED_IMAGE", "The file is not a valid image or is corrupted.")

    if fmt not in settings.ALLOWED_FORMATS:
        raise APIError(415, "UNSUPPORTED_FILE_TYPE", f"Image format {fmt} is not supported. Use PNG or JPEG.")
    if min(width, height) < settings.MIN_IMAGE_SIDE:
        raise APIError(422, "IMAGE_TOO_SMALL",
                       f"Image is {width}x{height}; each side must be at least {settings.MIN_IMAGE_SIDE} pixels.")
    if width * height > settings.MAX_IMAGE_PIXELS:
        raise APIError(413, "IMAGE_TOO_LARGE",
                       f"Image is {width}x{height} ({width * height / 1e6:.0f} MP); the limit is "
                       f"{settings.MAX_IMAGE_PIXELS / 1e6:.0f} MP.")
    return fmt, width, height


def validate_upload(filename, data):
    """Run all checks. Returns the decoded RGB image (numpy array) ready for the model."""
    check_filename(filename)
    check_size(data)
    check_header(data)
    try:
        return decode_image_bytes(data)          # step 7: full decode (catches truncated files)
    except ValueError:
        raise APIError(422, "CORRUPTED_IMAGE", "The image could not be decoded; the file may be corrupted.")
