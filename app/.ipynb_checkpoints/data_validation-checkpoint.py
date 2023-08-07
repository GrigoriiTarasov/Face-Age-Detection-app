

import io
import numpy as np
from PIL import Image
from fastapi import File

async def read_imagefile(file) -> np.ndarray:
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        return image_np
    except error as e:
        raise  e

async def handle_input_file(file):
    suitable_extensions = ("jpg", "jpeg", "png")
    recieved_extenion = file.filename.split(".")[-1]
    
    extension_valid = recieved_extenion in suitable_extensions
    if not(extension_valid):
         raise Exception(f'Wrong file extension: {recieved_extenion}. Please convert to any of {suitable_extensions}') 
    
    else:
        return await read_imagefile(file)
    


