from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.layers.core import Flatten, Dense
from keras.models import Sequential
from helpers import resize_to_fit
from keras.layers import Dropout
from imutils import paths
from pathlib import Path
import numpy as np
import pickle
import cv2


char_images_folder = "extracted_chars_real"
model_filename = "models/captcha_model_real_data.hdf5"
model_labels_filename = "labels.dat"


data = []
labels = []

for image_file in paths.list_images(char_images_folder):

    image = cv2.imread(image_file)
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to 20x20
    image = resize_to_fit(image, 20, 20)

    # Add a third channel dimension to the image to make Keras happy
    image = np.expand_dims(image, axis=2)

    # The label is the name of the folder storing the number
    label = str(Path(image_file).parent)[-1]

    data.append(image)
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.10, random_state=0)

# Convert the labels into one-hot encodings
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
with open(model_labels_filename, "wb") as f:
    pickle.dump(lb, f)

model = Sequential()

model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(500, activation="relu"))

# Output layer with 10 nodes (one for each class)
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=8, epochs=10, verbose=1)

model.save(model_filename)
