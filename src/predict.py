"""Sample prediction with the trained DenseNet121.

The `DRPredictor` class is also what the backend will use, so a prediction made here
and one made through the API go through exactly the same code.

Command line:
    python -m src.predict --model models/densenet121_dr.keras --image path/to/fundus.png
"""

import argparse
import json

import numpy as np
from tensorflow import keras

from src import config
from src.preprocessing import decode_image_bytes, load_image, preprocess_image


class DRPredictor:
    """Loads the model once and predicts DR severity for single images.

    Output of every predict_* method:
        {
          "class_id": 2,
          "class_name": "Moderate",
          "confidence": 0.81,                    # probability of the predicted class
          "probabilities": {"No DR": 0.02, ...}  # all 5 classes, sum = 1
        }
    """

    def __init__(self, model_path, use_ben_graham=config.USE_BEN_GRAHAM):
        self.model = keras.models.load_model(model_path)
        self.img_size = self.model.input_shape[1]        # read from the model, e.g. 224
        self.use_ben_graham = use_ben_graham

    def predict_rgb(self, img):
        """img: RGB uint8 array (H, W, 3) of any size."""
        x = preprocess_image(img, self.img_size, self.use_ben_graham)
        batch = np.expand_dims(x.astype("float32"), axis=0)       # (1, 224, 224, 3)
        probs = self.model.predict(batch, verbose=0)[0]           # (5,)
        class_id = int(np.argmax(probs))
        return {
            "class_id": class_id,
            "class_name": config.CLASS_NAMES[class_id],
            "confidence": round(float(probs[class_id]), 4),
            "probabilities": {name: round(float(p), 4) for name, p in zip(config.CLASS_NAMES, probs)},
        }

    def predict_path(self, path):
        return self.predict_rgb(load_image(path))

    def predict_bytes(self, data):
        return self.predict_rgb(decode_image_bytes(data))


def main():
    parser = argparse.ArgumentParser(description="Predict DR severity for one fundus image")
    parser.add_argument("--model", default=f"models/{config.MODEL_FILENAME}")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    result = DRPredictor(args.model).predict_path(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
