import os
import numpy as np
import cv2
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
model = load_model('model.h5')

# Define the emotion labels
label_map = ['angry', 'neutral', 'fear', 'happy', 'sad', 'surprise']

# Path to test dataset
test_dir = r"C:\Users\somae\OneDrive\Desktop\emotion-based-music-ai\test_dataset"

# Initialize true and predicted labels
y_true = []
y_pred = []

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))  # Resize to 48x48
    img = img / 255.0  # Normalize
    img = np.reshape(img, (1, 48, 48, 1))  # Reshape for the model
    return img

# Loop through each emotion folder
for emotion in label_map:
    folder_path = os.path.join(test_dir, emotion)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        predicted_label = np.argmax(prediction)
        y_true.append(label_map.index(emotion))
        y_pred.append(predicted_label)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_map))
