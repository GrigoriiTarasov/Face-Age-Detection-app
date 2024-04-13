"""
provides preproc_img for Deepface embedding models directly by architecture name. Was tested for [Facenet', 'Facenet512', 'ArcFace', 'VGGFace'] """

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
from typing import Union

from deepface.commons.functions import normalize_input


def get_normalizer(architecture_name: str) -> str:
    """deepface has different normilizer_name for different tf
    but we want to run with appropriate by architecture_name"""
    normilizer_name = architecture_name
    if architecture_name == "VGGFace":
        tf_version = int(tf.__version__.split(".", maxsplit=1)[0])
        if tf_version == 2:
            normilizer_name += "2"
    elif architecture_name == "Facenet512":
        normilizer_name = "Facenet"

    return normilizer_name


class PreprocDeepface:
    """Encapsulate all steps prior to model"""

    def __init__(
        self, deepface_architecture_name: str, target_size: Union[tuple, list]
    ):

        self.target_size = target_size
        self.deepface_normilizer = get_normalizer(deepface_architecture_name)

    def preproc_img(self, img: "Convirtable to np.array") -> np.array:
        """Wraps deepace normalization with model input restriction and
        img transforms passing extract_face"""

        img = image.array_to_img(img)

        img = img.resize(self.target_size)
        x = image.img_to_array(img)

        x = x[..., ::-1]
        x = np.expand_dims(x, axis=0)  # to batch

        x = x / 255  # deepface specific
        x = normalize_input(x, normalization=self.deepface_normilizer)

        return x

    def deprocess_image(self, vggface_image):
        """
        VGG only to test if img==preproc_img(img)
        """

        x_temp = np.copy(vggface_image)

        x_temp[..., 2] += 131.0912
        x_temp[..., 1] += 103.8827
        x_temp[..., 0] += 91.4953
        x_temp = x_temp[..., ::-1]

        image = x_temp.astype(np.uint8)

        return image

    def preproc_by_path(
        self, img_path: str, target_size: Union[tuple, list]
    ) -> np.array:
        """
        Example:

        img_name= '81_1_0_20170120134927295.jpg.chip.jpg'
        img_path = f'{abs_img_folder}/{img_name}'
        img = load_img(img_path, target_size = TARGET_SIZE)
        nn_input_example = preproc_img(img)
        nn_input_example
        """
        img = load_img(img_path, target_size=target_size)
        nn_input_example = self.preproc_img(img)
        return nn_input_example
