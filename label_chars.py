from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
from pathlib import Path
import numpy as np
import imutils
import shutil
import pickle
import random
import cv2
import os


model_filename = "models/captcha_model_98.hdf5"
model_labels_filename = "labels.dat"
image_folder = "trainset_chars/"


with open(model_labels_filename, "rb") as f:
    lb = pickle.load(f)


model = load_model(model_filename)

captcha_images = list(paths.list_images(image_folder))


num_correct = 0
incorrect = []

for image_file in captcha_images:

    correct_label = str(Path(image_file).parent)[-1]

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    predictions = []

    char_image = resize_to_fit(image, 20, 20)
    char_image = np.expand_dims(char_image, axis=2)
    char_image = np.expand_dims(char_image, axis=0)

    prediction = model.predict(char_image)

    # Convert one hot to int
    char = lb.inverse_transform(prediction)[0]
    predictions.append(char)

    predicted_label = "".join(predictions)

    filename  = Path(image_file).stem + Path(image_file).suffix

    src = image_file
    dest = "trainset_chars/labeled/"
    if not os.path.exists(dest):
        os.mkdir(dest)

    for num in range(10):
        if int(predicted_label) == num:
            dest = dest + str(num)

            if not os.path.exists(dest):
                os.mkdir(dest)

            dest = shutil.move(src, dest)
