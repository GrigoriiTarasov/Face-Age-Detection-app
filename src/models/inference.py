
# load actual model configs
typical_dataset_im_size = (200,200)
best_detection_backend = 'mtcnn'

def extract_face(img_or_its_path,
                 typical_dataset_im_size,
                 enforce_detection):
    
    face_obj = DeepFace.extract_faces(img_or_its_path, #img_path, 
                                      detector_backend = backend,
                                      grayscale = False,
                                      align = False,
                                      enforce_detection=enforce_detection,
                                      target_size=typical_dataset_im_size)
    return face_obj
        
        
def esteem_age():
    pass
    
def recognition_pipeline(img_batch):
    img_or_its_path = img_batch
    
    outs = extract_face(img_or_its_path,
                 typical_dataset_im_size,
                 enforce_detection=False)
    
    return outs 
    '''
    
    faces=outs.image
    ages = []
    for face in faces:
        age = esteem_age(face)
    
    return outs, ages
    '''
    
    