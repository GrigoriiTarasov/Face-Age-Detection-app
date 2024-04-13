import numpy as np
from typing import Union, Type, List, Dict

from age_module.src.models.inference import extract_face


def test_extract_face():
    # Mock input data
    file_name = "116_1_0_20170120134921760.jpg.chip.jpg"
    img_path = f"./age_module/tests/data/{file_name}"
    enforce_detection = True
    typical_dataset_im_size = (224, 224)  # Example size
    detector_backend = "opencv"  # Example backend

    # Call the function
    result = extract_face(
        img_path, enforce_detection, typical_dataset_im_size, detector_backend
    )

    # Assertion
    assert isinstance(result, list)  # Ensure the return type is a list
    assert all(
        isinstance(face, dict) for face in result
    )  # Ensure each element is a dictionary
    assert all(
        "face" in face for face in result
    )  # Ensure each dictionary contains "face" key
    assert all(
        isinstance(face["face"], np.ndarray) for face in result
    )  # Ensure "face" value is a numpy array
    # Add more assertions based on the expected output structure and behavior of the function
