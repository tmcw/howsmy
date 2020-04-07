from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
from pathlib import Path
import numpy as np
import imutils
import pickle
import cv2


model_filename = "models/captcha_model_real_data.hdf5"
model_labels_filename = "labels.dat"
test_image_folder = "test_set"


with open(model_labels_filename, "rb") as f:
    lb = pickle.load(f)


model = load_model(model_filename)

captcha_images = list(paths.list_images(test_image_folder))
# captcha_images = np.random.choice(captcha_images, size=(10,), replace=False)

num_correct = 0
incorrect = []

for image_file in captcha_images:

    captcha_correct_text = Path(image_file).stem

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1] if imutils.is_cv3() else contours[0]

    char_image_regions = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if w / h > 1: # 1 - best value so far (hyperparam)
            half_width = int(w / 2)
            char_image_regions.append((x, y, half_width, h))
            char_image_regions.append((x + half_width, y, half_width, h))
        else:
            char_image_regions.append((x, y, w, h))

    char_image_regions = sorted(char_image_regions, key=lambda x: x[0])

    predictions = []

    for char_bounding_box in char_image_regions:
        x, y, w, h = char_bounding_box

        char_image = image[y:y + h, x:x + w]
        char_image = resize_to_fit(char_image, 20, 20)

        char_image = np.expand_dims(char_image, axis=2)
        char_image = np.expand_dims(char_image, axis=0)

        prediction = model.predict(char_image)

        char = lb.inverse_transform(prediction)[0]
        predictions.append(char)

    captcha_text = "".join(predictions)
    print(f"ACTUAL : {captcha_correct_text}")
    print(f"CAPTCHA: {captcha_text}\n")

    if captcha_text == captcha_correct_text:
        num_correct += 1
    else:
        incorrect.append(image_file)

print(f"Total captchas: {len(captcha_images)}")
print(f"Total correct: {num_correct}")

print(f"Real captcha accuracy: {(num_correct / len(captcha_images)) * 100}% \n")

print(f"Incorrect: \n{incorrect}")
