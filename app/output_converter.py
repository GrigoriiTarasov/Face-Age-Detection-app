
def extract_to_dct(face_obj:dict)->dict:
    '''Remove face:np.array from dict with prediction for 1 face to output only
    boundaries and confidence'''
    
    #for face_obj in extraction_obj:
    #    print(f'face_obj {face_obj}')
    del face_obj['face']
        
    return face_obj

def age_to_dct(age:int)->dict:
    '''Wraps age:int with dict to ouput jsonable'''
    return {'age':age}