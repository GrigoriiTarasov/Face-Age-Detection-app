#!/usr/bin/env python
# coding: utf-8
__doc__='''
uvicorn app.main:app --reload --port 8000
'''
# In[1]:


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import sys


sys.path.append('../')

app = FastAPI()



from src.models.inference import PipelinePredictor
#from src.models.inference import recognition_pipeline


predictor = PipelinePredictor()


@app.on_event('startup')
def prepare_pipeline():
    print("Loading model...")
    predictor.load_model()
    print("Loaded succesivly...")
    #return model

@app.get("/ping")
async def ping():
    return "Hello, I am alive"


# # Core

from data_validation import handle_input_file

@app.post("/api/predict_photo")
async def predict(
    file: UploadFile = File(...)
):
    try:
        image_np = handle_input_file(file)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f'Error: {e}'})

    
    #image = read_file_as_image(await file.read())
    #img_batch = np.expand_dims(image, 0)
    #predictions = Predictor.model.predict(img_batch)
    #print(predictions[0])
    #predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    #confidence = np.max(predictions[0])
    
    #outs =  recognition_pipeline(file)
    prediction = {}
    
    return JSONResponse(status_code=200, content={"message": prediction})


# In[ ]:


@app.post("/api/predict_video")
async def predict(
    file: UploadFile = File(...)
):
    
    Predictor.model
    
    img_batch = np.expand_dims(image, 0)
    
    predictions = Predictor.model.predict(img_batch)
    
    
    #print(predictions[0])
    #predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    #confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


# In[ ]:


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)



