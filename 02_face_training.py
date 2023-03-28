import cv2
import numpy as np
import os

# Set the current working directory to the directory containing the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Path for face image database
path = r'C:\Users\Harshit Singh\Downloads\DC\lfw\lfw'

# Load all the images
face_images = []
labels = []

for dir_name in os.listdir(path):
    dir_path = os.path.join(path, dir_name)
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            if filename.endswith('.jpg'):
                # Load the image and convert to grayscale
                img = cv2.imread(os.path.join(dir_path, filename))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Extract the integer part of the label from the filename
                label = int(filename.split('_')[-1].split('.')[0])

                # Add the image and corresponding label to the training data
                face_images.append(gray)
                labels.append(label)

# Train the face recognition model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_images, np.asarray(labels))

# Save the trained model
face_recognizer.save('trained_model.xml')
print('Model trained and saved successfully!')
