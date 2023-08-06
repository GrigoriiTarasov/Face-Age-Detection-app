
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


def get_params_of_model(configs_path):
    pass

class PipelinePredictor:
    '''Loads weights and archs using yaml
    default is best
    '''
    def __init__(self, 
                 configs_path = '../'):
        '''
        Default configs_path is best_configs_path
        '''
        self.model = None
        self.age_params = get_params_of_model(configs_path)
        
    def load_model(self):
        '''
        '''
        #self.age_params
        self.model = None
    
    
def esteem_age(age_model):
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
    
    