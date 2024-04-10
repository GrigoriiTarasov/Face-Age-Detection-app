#!/usr/bin/env python
# coding: utf-8
__doc__ = """
uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import sys
from typing import List, Dict
import uvicorn

sys.path.append("../")

from age_module.app.data_validation import handle_input_pic, handle_input_video
from age_module.src.models.inference import PipelinePredictor
from age_module.app.output_converter import AdaptPredAPI


app = FastAPI()
predictor = PipelinePredictor()

def predict_one_frame(frame: np.array, 
                      predictor: PipelinePredictor) -> List[Dict]:

    preds = predictor.predict_img(frame)
    client_output = [AdaptPredAPI(*pred).client_output() for pred in preds]

    return client_output


# # Web


@app.on_event("startup")
def prepare_pipeline():
    print("Loading model...")
    predictor.load_model()
    print("Loaded succesivly...")
    # return model


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


# # Core


@app.post("/api/predict_photo")
async def predict_photo(file: UploadFile = File(...)):
    try:
        image_np = await handle_input_pic(file)
    except Exception as e:
        return JSONResponse(
            status_code=400, content={"message": f"Error in input data: {e}"}
        )

    prediction = predict_one_frame(image_np, predictor)

    return JSONResponse(status_code=200, content=prediction)


@app.post("/api/predict_video")
async def predict_video(file: UploadFile = File(...)):
    try:
        frames = await handle_input_video(file)
    except Exception as e:
        return JSONResponse(
            status_code=400, content={"message": f"Error in input data: {e}"}
        )

    predictions = []
    for i, frame in enumerate(frames):
        frame_dict = dict(frame_ind=i)
        prediction = predict_one_frame(frame, predictor)
        predictions.append({**frame_dict, **dict(prediction=prediction)})

    print(predictions)
    return JSONResponse(status_code=200, content=predictions)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
