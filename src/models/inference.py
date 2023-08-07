
import numpy as np
import sys
from tensorflow.keras.models import load_model
import tensorflow as tf
from typing import List, Dict

sys.path.append('../deepface/')
from deepface import DeepFace

sys.path.append('../')
from src.models import tf_mem
import src.models.img_utils as img_utils

# load actual model configs
typical_dataset_im_size = (200,200)
best_detection_backend = 'mtcnn'
gpu_gb_limit = 6

age_best_model_path = '../weights/Facenet_121_age_checkpoint_best.h5'
age_base_model_name = 'Facenet'
target_size = (160, 160)
preproc_deepface = img_utils.PreprocDeepface(age_base_model_name, target_size)

tf_mem.limit_gpu_gb(gpu_gb_limit)

def extract_face(img_or_its_path,
                 enforce_detection:bool,
                 typical_dataset_im_size:tuple=typical_dataset_im_size,
                 detector_backend:str=best_detection_backend)->List[Dict]:
    
    face_obj = DeepFace.extract_faces(img_or_its_path, #img_path, 
                                      detector_backend = detector_backend,
                                      grayscale = False,
                                      align = False,
                                      enforce_detection=enforce_detection,
                                      target_size=typical_dataset_im_size)
    return face_obj


def get_params_of_model(configs_path):
    pass

class PipelinePredictor:
    '''Loads weights and archs using yaml
    default is best
    '''
    def __init__(self, 
                 configs_path = age_best_model_path):
        '''
        Default configs_path is best_configs_path
        '''
        self.age_model = load_model(age_best_model_path)
        #self.age_params = get_params_of_model(configs_path)
        
    '''def load_model(self):
         
        #self.age_params
        self.model = None
    '''
    def predict_img(self, img:np.array):
        
        extraction_obj, age_pred = recognition_pipeline(img)
        
        return extraction_obj, age_pred
    
    
def esteem_age(img_batch,
               age_model:tf.keras.Model,
               preproc_f)->float:
    
    img_np = preproc_f(img_batch)
    prediction = age_model.predict(img_np)
    return prediction 

def uniting_age(age_pred:float)->int:
    '''Prediction to base units int can be done using '''
    return int(age_pred)

    
def recognition_pipeline(img_batch:np.array,
                         age_model:tf.keras.Model,
                         age_preproc):
    
    img_or_its_path = img_batch
    extraction_obj = extract_face(img_or_its_path,
                        enforce_detection=False,
                        typical_dataset_im_size=typical_dataset_im_size,
                        detector_backend = best_detection_backend
                 )
    
    age_preds = []
    for face_obj in extraction_obj:
        
        age_pred = inference.esteem_age(face_obj['face'],
                                        age_model,
                                        age_preproc) # preproc_deepface.preproc_img
        age_preds.append(age_pred[0][0]) # array([[84.32]], dtype=float32)
        
    
    return extraction_obj, age_preds
    '''
    
    faces=outs.image
    ages = []
    for face in faces:
        age = esteem_age(face)
    
    return outs, ages
    '''
    
    