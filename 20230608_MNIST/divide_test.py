import numpy as np
from PIL import Image

def imageToData(imagename):
    greyImage = Image.open(imagename).convert("L")
    greyImage = greyImage.resize((8, 8), Image.Resampling.LANCZOS)
    
    numImage = numpy.asarray(greyImage, dtype = float)
    numImage = 16 - numpy.floor(17 * numImage / 256)
    numImage = numImage.flatten()
    return numImage