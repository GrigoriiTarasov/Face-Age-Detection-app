import numpy as np
import os
import sys
from tensorflow.keras.models import load_model
import tensorflow as tf
from typing import List, Dict, Union, Type
import yaml

from deepface import DeepFace

from age_module.src.models import tf_mem
import age_module.src.models.img_utils as img_utils


# fix for pytest to work
age_module_dir = os.path.dirname(
    os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
)
sys.path.append(age_module_dir)
configs_rel_path = os.path.join(age_module_dir, "../configs")

# Constants
INFER_CONFIG_PATH = f"{configs_rel_path}/infer.yaml"
DETECTION_CONFIG_PATH = f"{configs_rel_path}/detection.yaml"
AGE_ESTIMATION_CONFIG_PATH = f"{configs_rel_path}/age_esteem.yaml"

# Load confiurations
with open(INFER_CONFIG_PATH) as f:
    infer_dct = yaml.safe_load(f)
device_type = infer_dct["device_type"]
gpu_gb_limit = infer_dct["gpu_gb_limit"]

with open(DETECTION_CONFIG_PATH) as f:
    detection_dct = yaml.safe_load(f)
best_detection_backend = detection_dct["best_detection_backend"]
typical_dataset_im_size = detection_dct["typical_dataset_im_size"]

with open(AGE_ESTIMATION_CONFIG_PATH) as f:
    age_esteem_dct = yaml.safe_load(f)
age_best_model_path = age_esteem_dct["age_best_model_path"]
age_preproc_family = age_esteem_dct["age_preproc_family"]
target_size = age_esteem_dct["target_size"]


preproc_deepface = img_utils.PreprocDeepface(age_preproc_family, target_size)

tf_mem.limit_gpu_gb(gpu_gb_limit)


def extract_face(
    img_or_its_path: Union[Type[np.ndarray], str],
    enforce_detection: bool,
    typical_dataset_im_size: tuple = typical_dataset_im_size,
    detector_backend: str = best_detection_backend,
) -> List[Dict]:
    """
    Extracts faces from an image.

    Args:
        img_or_its_path (Union[str, np.ndarray]): Image path or numpy array.
        enforce_detection (bool): Whether to enforce face detection.
        typical_dataset_im_size (tuple, optional): Typical dataset image size.
        detector_backend (str, optional): Face detection backend.

    Returns:
        List[Dict]: List of face objects.
    """
    face_obj = DeepFace.extract_faces(
        img_or_its_path,
        detector_backend=detector_backend,
        grayscale=False,
        align=False,
        enforce_detection=enforce_detection,
        target_size=typical_dataset_im_size,
    )
    return face_obj


def get_params_of_model(configs_path):
    pass


class PipelinePredictor:
    """Loads weights and archs using yaml
    default is best
    """

    def __init__(self, preproc_deepface=preproc_deepface):
        """
        Default configs_path is best_configs_path
        """
        self.age_model = None
        self.preproc_deepface = preproc_deepface

    def load_model(self, age_model_path=age_best_model_path):
        self.age_model = load_model(age_model_path)

    def predict_img(self, img: np.array):

        extraction_obj, age_pred = recognition_pipeline(
            img, self.age_model, self.preproc_deepface.preproc_img
        )

        return extraction_obj, age_pred


def esteem_age(img_batch, age_model: tf.keras.Model, preproc_f) -> float:

    img_np = preproc_f(img_batch)
    prediction = age_model.predict(img_np)
    return prediction


def age_postprocess(age_pred: Union[Type[np.ndarray], float, list], max_age=116) -> int:
    """Prediction to base units (int) and not to extropolate"""
    predicted_age = np.clip(np.round(age_pred * max_age), 1, max_age)
    predicted_age = int(predicted_age)

    return predicted_age


def recognition_pipeline(
    img_batch: np.array,
    age_model: tf.keras.Model,
    age_preproc: img_utils.PreprocDeepface,
    age_raw: bool = False,
):
    """Extracts face from image and gets age prediction over it"""

    img_or_its_path = img_batch
    extraction_obj = extract_face(
        img_or_its_path,
        enforce_detection=False,
        typical_dataset_im_size=typical_dataset_im_size,
        detector_backend=best_detection_backend,
    )

    age_preds = []
    for face_obj in extraction_obj:

        age_pred = esteem_age(face_obj["face"], age_model, age_preproc)

        if not age_raw:
            age_pred = age_postprocess(age_pred)

        age_preds.append(age_pred)  # array([[84.32]], dtype=float32)

    return extraction_obj, age_preds
