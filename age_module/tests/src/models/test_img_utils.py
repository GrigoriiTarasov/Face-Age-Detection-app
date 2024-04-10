import pytest
import numpy as np
from PIL import Image

from age_module.src.models.img_utils import PreprocDeepface


# Fixture for PreprocDeepface instance
@pytest.fixture
def preproc_instance():
    return PreprocDeepface(deepface_architecture_name="VGGFace", target_size=(224, 224))


# Test case for preproc_img method
def test_preproc_img(preproc_instance):
    # Create a sample image
    img = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    # Preprocess the image
    processed_img = preproc_instance.preproc_img(img)
    # Check if the output is a numpy array
    assert isinstance(processed_img, np.ndarray)
    # Check if the output has the correct shape
    assert processed_img.shape == (1, 224, 224, 3)


# Test case for deprocess_image method
def test_deprocess_image(preproc_instance):
    # Create a sample image
    img = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    # Preprocess the image
    processed_img = preproc_instance.preproc_img(img)
    # Deprocess the processed image
    deprocessed_img = preproc_instance.deprocess_image(processed_img[0])
    # Check if the output is a numpy array
    assert isinstance(deprocessed_img, np.ndarray)
    # Check if the output has the correct shape
    assert deprocessed_img.shape == (224, 224, 3)


# Test case for preproc_by_path method
def test_preproc_by_path(preproc_instance):
    # Path to a sample image
    img_path = "sample_image.jpg"
    # Create a sample image
    img = Image.new("RGB", (224, 224))
    img.save(img_path)
    # Preprocess the image using the method
    processed_img = preproc_instance.preproc_by_path(img_path, target_size=(224, 224))
    # Check if the output is a numpy array
    assert isinstance(processed_img, np.ndarray)
    # Check if the output has the correct shape
    assert processed_img.shape == (1, 224, 224, 3)

    # Clean up - remove the sample image
    import os

    os.remove(img_path)


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
