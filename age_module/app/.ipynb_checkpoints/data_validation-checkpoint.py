import cv2
from fastapi import File
import io
import numpy as np
from PIL import Image
import tempfile


async def read_imagefile(file) -> np.ndarray:
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        return image_np
    except error as e:
        raise e


async def handle_input_pic(file):
    """Check that photo"""
    suitable_extensions = ("jpg", "jpeg", "png")
    recieved_extenion = file.filename.split(".")[-1]

    extension_valid = recieved_extenion in suitable_extensions
    if not (extension_valid):
        raise Exception(
            f"Wrong file extension: {recieved_extenion}. Please convert to any of {suitable_extensions}"
        )

    else:
        return await read_imagefile(file)


async def handle_input_video(file):
    """Check and process video frames"""
    suitable_extensions = ("mp4", "avi", "mov")  # Add supported video extensions here
    received_extension = file.filename.split(".")[-1]

    extension_valid = received_extension in suitable_extensions
    if not extension_valid:
        raise Exception(
            f"Wrong file extension: {received_extension}. Please convert to any of {suitable_extensions}"
        )

    else:
        return await read_video_frames(file)


async def read_video_frames(file) -> list:
    """Read video frames using OpenCV"""
    try:
        video_bytes = await file.read()
        frames = []

        # Create a temporary file and write the video bytes to it
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(video_bytes)

        cap = cv2.VideoCapture(temp_file.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        return frames
    except Exception as e:
        raise e
