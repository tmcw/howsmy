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


model_filename = "captcha_model_fake_data.hdf5"
model_labels_filename = "labels.dat"
image_folder = "/Users/oliverproud/howsmy-working/extracted_chars_test/"


with open(model_labels_filename, "rb") as f:
    lb = pickle.load(f)


model = load_model(model_filename)

captcha_images = list(paths.list_images(image_folder))
# captcha_images = np.random.choice(captcha_images, size=(10,), replace=False)


# This function renames a file in the destination path
# if the file that's being moved to the destination already exists
def rename_file(dest, filename):
    if os.path.exists(dest + "/" + filename):
        os.rename(dest + "/" + filename, dest + "/" + new_name + ".png")


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
    new_name = str(random.randint(5000, 1000000))

    if int(predicted_label) == 0:
        src = image_file
        dest = "/Users/oliverproud/howsmy-working/extracted_chars_test/test/0"

        rename_file(dest, filename)

        dest = shutil.move(src, dest)

    elif int(predicted_label) == 1:
        src = image_file
        dest = "/Users/oliverproud/howsmy-working/extracted_chars_test/test/1"

        rename_file(dest, filename)

        dest = shutil.move(src, dest)

    elif int(predicted_label) == 2:
        src = image_file
        dest = "/Users/oliverproud/howsmy-working/extracted_chars_test/test/2"

        rename_file(dest, filename)

        dest = shutil.move(src, dest)

    elif int(predicted_label) == 3:
        src = image_file
        dest = "/Users/oliverproud/howsmy-working/extracted_chars_test/test/3"

        rename_file(dest, filename)

        dest = shutil.move(src, dest)

    elif int(predicted_label) == 4:
        src = image_file
        dest = "/Users/oliverproud/howsmy-working/extracted_chars_test/test/4"

        rename_file(dest, filename)

        dest = shutil.move(src, dest)

    elif int(predicted_label) == 5:
        src = image_file
        dest = "/Users/oliverproud/howsmy-working/extracted_chars_test/test/5"

        rename_file(dest, filename)

        dest = shutil.move(src, dest)

    elif int(predicted_label) == 6:
        src = image_file
        dest = "/Users/oliverproud/howsmy-working/extracted_chars_test/test/6"

        rename_file(dest, filename)

        dest = shutil.move(src, dest)

    elif int(predicted_label) == 7:
        src = image_file
        dest = "/Users/oliverproud/howsmy-working/extracted_chars_test/test/7"

        rename_file(dest, filename)

        dest = shutil.move(src, dest)

    elif int(predicted_label) == 8:
        src = image_file
        dest = "/Users/oliverproud/howsmy-working/extracted_chars_test/test/8"

        rename_file(dest, filename)

        dest = shutil.move(src, dest)

    elif int(predicted_label) == 9:
        src = image_file
        dest = "/Users/oliverproud/howsmy-working/extracted_chars_test/test/9"

        rename_file(dest, filename)

        dest = shutil.move(src, dest)
