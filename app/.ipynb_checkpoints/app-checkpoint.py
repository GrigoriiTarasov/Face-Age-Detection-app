#!/usr/bin/env python
# coding: utf-8
__doc__='''
uvicorn app.main:app --reload --port 8000
'''
# In[1]:


from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import sys


sys.path.append('../')

app = FastAPI()


# In[ ]:


from src.models import load_model


# In[ ]:


class Predictor:
    def __init__(self):
        model = None


# In[ ]:


@app.on_event('startup')
def init_data():
    print("Loading model...")
    Predictor.model = load_model()
    print("Loaded succesivly...")
    return model


# In[ ]:


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


# # Core

# In[ ]:


from src.models.inference import recognition_pipeline


# In[ ]:


@app.post("/api/predict_photo")
async def predict(
    file: UploadFile = File(...)
):
    #image = read_file_as_image(await file.read())
    #img_batch = np.expand_dims(image, 0)
    #predictions = Predictor.model.predict(img_batch)
    #print(predictions[0])
    #predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    #confidence = np.max(predictions[0])
    
    outs =  recognition_pipeline(file)
    
    return outs


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


# In[ ]:





# In[ ]:





# In[ ]:




