import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from image_to_vector import image_to_vector
import os

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax', ),
])

# Executing setup.py and creates model, if it doesn't exist
if os.path.exists("model.h5") == False:
    print("Model doesn't exist, let me create one")
    with open('setup.py', 'r') as file:
        setup = file.read()
    
    exec(setup)

# Load saved weights into the model.
model.load_weights('model.h5')

# Use function to change image to vector
data = image_to_vector('digit500x500.png')

# Get prediction from model
prediction = model.predict(data)

# Change it to readable result and print
print(np.argmax(prediction, axis=1)[0])