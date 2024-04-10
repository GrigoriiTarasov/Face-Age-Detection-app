class AdaptPredAPI:
    '''Encapsulates convertion of models outputs to app
    API for client'''
    def __init__(self, face_obj : dict, age : int):
        self.face_obj = face_obj
        self.age = age

    def client_output(self) -> dict:
        # select only output keys
        self.extract_to_dct()
        # unite all models outputs to clients jsonable output
        return {
              **self.face_obj,
              **self.age_to_dct(),
            }
    
    def extract_to_dct(self) -> None:
        """Convert dict like return of DeepFace.extract_faces  to 
        API output format of boundaries and confidence"""

        # self prevents TypeError: 'AdaptPredAPI' object does not support item dele...
        del self.face_obj["face"]
    
    def age_to_dct(self) -> dict:
        """Wraps age:int with dict to make ouput jsonable"""
        return {"age": self.age}
