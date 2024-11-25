import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations and working with arrays
import cv2 as cv  # For image processing tasks
import os  # For interacting with the operating system, like file paths
import tensorflow as tf  # For building and training neural network models

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model  # For loading a saved Keras model
from keras.models import Sequential  # For creating a linear stack of layers in the model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten  # For building model layers
from keras.optimizers import Adam  # For optimization algorithms
from keras.layers import BatchNormalization  # For applying Batch Normalization in neural network layers
from keras.regularizers import l2  # For applying L2 regularization to prevent overfitting
from keras.callbacks import ReduceLROnPlateau, EarlyStopping  # Importing specific callback functions

import warnings  # For handling warnings
import sys  # For interacting with the Python interpreter
if not sys.warnoptions:
    warnings.simplefilter("ignore")  # Ignore simple warnings if not already done
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignore deprecation warnings

import shutil

# Define the path to your original dataset and the paths where you want to store your train and test datasets
original_dataset_dir = r'C:\Users\somae\OneDrive\Desktop\emotion-based-music-ai\dataset'
train_dir = r'C:\Users\somae\OneDrive\Desktop\emotion-based-music-ai\dataset\train_data'
test_dir = r'C:\Users\somae\OneDrive\Desktop\emotion-based-music-ai\dataset\test_data'

# Create directories for training and testing datasets if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the split ratio
train_ratio = 0.8

# Loop through each emotion category in the original dataset
for emotion in os.listdir(original_dataset_dir):
    emotion_dir = os.path.join(original_dataset_dir, emotion)
    if os.path.isdir(emotion_dir):
        # Get a list of all the image filenames in the emotion category
        images = [f for f in os.listdir(emotion_dir) if os.path.isfile(os.path.join(emotion_dir, f))]
        
        # Randomly shuffle the list of image filenames
        np.random.shuffle(images)
        
        # Split the list of image filenames into training and testing sets
        train_size = int(len(images) * train_ratio)
        train_images = images[:train_size]
        test_images = images[train_size:]
        
        # Create directories for the emotion category in the train and test datasets
        train_emotion_dir = os.path.join(train_dir, emotion)
        test_emotion_dir = os.path.join(test_dir, emotion)
        os.makedirs(train_emotion_dir, exist_ok=True)
        os.makedirs(test_emotion_dir, exist_ok=True)
        
        # Copy the images into the corresponding directories
        for image in train_images:
            shutil.copy(os.path.join(emotion_dir, image), os.path.join(train_emotion_dir, image))
        for image in test_images:
            shutil.copy(os.path.join(emotion_dir, image), os.path.join(test_emotion_dir, image))

print("Dataset splitting complete")

# Image Preprocessing and Data Augmentation
train_data_generator = ImageDataGenerator(
    rescale=1./255,  # Rescale the pixel values (normalization)
    rotation_range=15,  # Random rotation in the range of 15 degrees
    width_shift_range=0.15,  # Random horizontal shifts (15% of total width)
    height_shift_range=0.15,  # Random vertical shifts (15% of total height)
    shear_range=0.15,  # Random shearing transformations
    zoom_range=0.15,  # Random zoom range
    horizontal_flip=True,  # Randomly flip inputs horizontally
)

fer_training_data = train_data_generator.flow_from_directory(
    r'C:\Users\somae\OneDrive\Desktop\emotion-based-music-ai\dataset\train_data',  # Path to the training data
    target_size=(48, 48),  # Resize images to 48x48
    batch_size=64,  # Number of images to yield per batch
    color_mode='grayscale',  # Load images in grayscale
    class_mode='categorical'  # Labels will be returned in categorical format
)

# Initialize an ImageDataGenerator for test data with rescaling
test_data_generator = ImageDataGenerator(rescale=1./255)

fer_test_data = test_data_generator.flow_from_directory(
    r'C:\Users\somae\OneDrive\Desktop\emotion-based-music-ai\dataset\test_data',  # Directory path for test images
    target_size = (48, 48),  # Resizes images to 48x48 pixels
    batch_size = 64,  # Number of images to yield per batch
    color_mode = 'grayscale',  # Specifies that images are in grayscale
    class_mode = 'categorical'  # Images are classified categorically
)

# Define the model architecture
model = Sequential()

model.add(Conv2D(filters=512, kernel_size=(5, 5), input_shape=(48, 48, 1), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_1'))
model.add(BatchNormalization(name='batchnorm_1'))
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_2'))
model.add(BatchNormalization(name='batchnorm_2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'))
model.add(Dropout(0.25, name='dropout_1'))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_3'))
model.add(BatchNormalization(name='batchnorm_3'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_4'))
model.add(BatchNormalization(name='batchnorm_4'))
model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'))
model.add(Dropout(0.25, name='dropout_2'))

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_5'))
model.add(BatchNormalization(name='batchnorm_5'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_6'))
model.add(BatchNormalization(name='batchnorm_6'))
model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3'))
model.add(Dropout(0.25, name='dropout_3'))

model.add(Flatten(name='flatten'))
model.add(Dense(256, activation='elu', kernel_initializer='he_normal', name='dense_1'))
model.add(BatchNormalization(name='batchnorm_7'))
model.add(Dropout(0.25, name='dropout_4'))
model.add(Dense(7, activation='softmax', name='out_layer'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [early_stopping, lr_scheduler]

# Train the model
history = model.fit(
    fer_training_data,
    epochs=60, 
    validation_data=fer_test_data,
    batch_size=64,
    callbacks=callbacks,
)

# Save the model
model.save('model.h5')
print("Model saved as model.h5")

# Function to predict emotion and handle neutral case
def predict_emotion(image_path, model, labels, threshold=0.5):

    # Load the image and preprocess
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (48, 48))
    img = img.astype("float32") / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension

    # Get prediction probabilities
    prediction = model.predict(img)
    max_prob = np.max(prediction)
    
    # If the confidence is below the threshold, classify as neutral
    if max_prob < threshold:
        return "neutral"
    else:
        predicted_class = np.argmax(prediction)
        return labels[predicted_class]

# List of emotion labels in the same order as the model output
emotion_labels = ['anger', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Plotting training and validation accuracy
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, len(history.history['accuracy']) + 1)), y=history.history['accuracy'], mode='lines+markers', name='Training Accuracy'))
fig.add_trace(go.Scatter(x=list(range(1, len(history.history['val_accuracy']) + 1)), y=history.history['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))
fig.update_layout(title='Training vs. Validation Accuracy', xaxis_title='Epochs', yaxis_title='Accuracy')
fig.show()
