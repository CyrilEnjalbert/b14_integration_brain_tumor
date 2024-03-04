# Lire une image Base64

import base64
from PIL import Image
from io import BytesIO

def open_image_from_base64(base64_string):
    # Decode base64 string to bytes
    image_bytes = base64.b64decode(base64_string)
    
    # Open bytes as image using PIL
    image = Image.open(BytesIO(image_bytes))
    
    return image
