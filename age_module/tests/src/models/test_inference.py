import numpy as np
from typing import Union, Type, List, Dict
import pytest
from age_module.src.models.inference import (
    extract_face,
    age_postprocess,
    PipelinePredictor,
    esteem_age,
    recognition_pipeline,
)

weight_pah = r"./age_module/weights/Facenet_121_age_checkpoint_best.h5"


def test_extract_face():
    # Mock input data
    file_name = "116_1_0_20170120134921760.jpg.chip.jpg"
    img_path = f"./age_module/tests/data/{file_name}"
    enforce_detection = True
    typical_dataset_im_size = (224, 224)  # Example size
    detector_backend = "mtcnn"  # Example backend

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


def test_age_postprocess():
    # Mock input data
    ages_out = [-1, 500]
    max_age_out = 116
    results_of_out = age_postprocess(ages_out, max_age=max_age_out)
    assert all(results_of_out <= max_age_out)

    age_fractional = 0.3
    assert type(age_postprocess(age_fractional)) is int


class MockPreprocDeepface:
    def __call__(self, img):
        return img  # Placeholder for preprocessing function

    def preproc_img(self, img):
        return img  # Placeholder for preprocessing function


class MockModel:
    def predict(self, img_batch):
        return np.array([[30.0]])  # Placeholder for model prediction


@pytest.fixture
def mock_preproc_deepface():
    return MockPreprocDeepface()


@pytest.fixture
def mock_age_model():
    return MockModel()


def test_PipelinePredictor_load_model(mock_age_model):
    predictor = PipelinePredictor()
    predictor.load_model(weight_pah)
    assert predictor.age_model is not None


def test_esteem_age(mock_age_model, mock_preproc_deepface):
    img_batch = np.ones((10, 100, 100, 3))  # Placeholder batch of images
    prediction = esteem_age(
        img_batch, mock_age_model, mock_preproc_deepface.preproc_img
    )
    assert isinstance(prediction, np.ndarray)


def test_recognition_pipeline(mock_age_model, mock_preproc_deepface):
    # img_batch = np.ones((100, 100, 3))  # Placeholder batch of images
    file_name = "116_1_0_20170120134921760.jpg.chip.jpg"
    img_path = f"./age_module/tests/data/{file_name}"
    extraction_obj, age_preds = recognition_pipeline(
        img_path, mock_age_model, mock_preproc_deepface
    )
    assert len(extraction_obj) == 1
    assert isinstance(age_preds, list)
    assert isinstance(age_preds[0], np.ndarray)
    assert age_preds[0].shape == (1, 1)  # Assuming output shape of age prediction
