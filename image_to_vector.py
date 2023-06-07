from PIL import Image
import numpy as np

def image_to_vector(file):
    image = Image.open(file)
    image = image.convert("L").resize((28, 28))

    vector = np.asarray(image)
    vector = (vector / 255) - 0.5
    vector = vector.reshape((-1, 784))

    return vector