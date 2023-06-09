import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import os

# Fetch the data using mnist
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images (changes grayscale to form that the machine can understand)
test_images = (test_images / 255) - 0.5

# Flatten the images (merges multiple lists into one list)
test_images = test_images.reshape((-1, 784))

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax', ),
])

# if model.h5 exist, test functionality, if not generate one
if os.path.exists("model.h5") == False:
    # Train data and configuration
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    train_images = (train_images / 255) - 0.5
    train_images = train_images.reshape((-1, 784))

    # Configure the model
    model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

    # Train the model
    model.fit(
        train_images, # training data
        to_categorical(train_labels), # training targets
        epochs=5,
        batch_size=32,
    )

    # Test the model
    model.evaluate(
        test_images,
        to_categorical(test_labels)
    )

    # Save the model weights for later use in model.h5
    model.save_weights('model.h5')
else:
    # Load the model's saved weights.
    model.load_weights('model.h5')
    
    # Predict on the first 5 test images.
    predictions = model.predict(test_images[:5])

    # Print our model's predictions.
    print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

    # Check our predictions against the ground truths.
    print(test_labels[:5]) # [7, 2, 1, 0, 4]