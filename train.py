from tensorflow.keras import Input, Model
from tensorflow.keras.models import load_model
import toml
from tensorflow.keras.layers import (
    Conv2D,
    Activation,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import top_k_categorical_accuracy
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    LambdaCallback,
    TerminateOnNaN,
)
from os.path import isfile
from hashlib import blake2b
import tensorflow as tf
from generator.generator import Generator 

# tmcw: tweaked because my target is only numeric
alphabet = "0123456789"
char_size = len(alphabet)
categories = 5
model_path = "./checkpoints/model.hdf5"
h = 40
w = 120

class Opt(object):
  def __init__(self):
    self.isTune=  True
    self.epoch= 200
    self.lr= 0.0001
    self.train_size= 500
    self.loadHeight=40
    self.loadWidth=120
    self.cap_len =5
    self.char_set_len=10
    self.batchSize = 128

generator = Generator(Opt())
train_generator = generator.synth_generator('train')
val_generator = generator.synth_generator('val')


def text2vec(label):
    vecs = []
    for char in label:
        vector = np.zeros(char_size)
        vector[alphabet.index(char)] = 1
        vecs.append(vector)
    return np.array(vecs)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# def generator():
#     while True:
#         dataset = list(toml.load('./training.toml').items())
#         for chunk in chunks(dataset, 32):
#             if len(chunk) != 32:
#                 continue
#             x = []
#             y = []
# 
#             for (key, code) in chunk:
#                 file = open("./images/captcha-%s.png" % key.zfill(3), 'rb')
#                 img = img_to_array(Image.open(file).convert("L")).reshape((h, w, 1))
#                 img /= 127.5
#                 img -= 1.0
#                 x.append(img)
#                 y.append(text2vec(code))
# 
#             x = np.array(x)
#             # crashes here. reshape with original argument of 32 doesn't work, probably
#             # because there are only 10 numbers supported rather than 36 numbers + letters
#             # print("chunk len: %d, len: %d, shape: %s" % (len(chunk), len(y), np.array(y).shape))
#             y = np.array(y).reshape((32, -1))
#             if np.any(np.isnan(x)) or np.any(np.isnan(y)):
#                 print("OPOO got NAN")
#                 continue
#             yield (x, y)

def vec2text(label):
    arr = [l.argmax().tolist() for l in label.reshape((categories, char_size))]
    ret = [alphabet[l] for l in arr]
    return ret


class CaptchaModel:
    # ref: https://github.com/yeguixin/captcha_solver/tree/master/src/models
    def __init__(self):
        self.filewriter = tf.summary.create_file_writer("./logs/")
        if isfile(model_path):
            self.model = load_model(model_path)
            return
        inp = Input(shape=(h, w, 1))
        x = Conv2D(32, (3, 3), strides=(1, 1), padding="same")(inp)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        x = Dropout(0.5)(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        x = Dropout(0.5)(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        x = Dropout(0.5)(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        x = Dropout(0.5)(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(char_size * categories, activation="softmax")(x)
        model = Model(inp, x)
        model.compile(
            optimizer=RMSprop(learning_rate=0.0001, clipvalue=0.5),
            loss=["categorical_crossentropy"],
            metrics=[self.accuracy],
        )
        self.model = model

    def accuracy(self, y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=categories)

    def test(self, batch, logs=None):
        (images, labels) = next(val_generator)
        for i, image in enumerate(images[:5]):
            label = "".join(vec2text(labels[i].reshape((1, -1))))
            predicted = self.model.predict(image.reshape((1, h, w, 1)))
            predicted_label = "".join(vec2text(predicted))
            print(
                blake2b(image.tobytes()).hexdigest()[:5],
                predicted[0][:5],
                vec2text(predicted),
                vec2text(labels[i].reshape((1, -1))),
            )
            with self.filewriter.as_default():
                tf.summary.image(
                    f"{label} {predicted_label}", image.reshape((1, h, w, 1)), step=0
                )

    def train(self):
        checkpoint = ModelCheckpoint(
            model_path, save_best_only=True, monitor="accuracy", mode="max"
        )
        board = TensorBoard(log_dir="./logs", write_images=True)
        log = LambdaCallback(on_epoch_end=self.test)
        term = TerminateOnNaN()
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=20,
            epochs=1000,
            callbacks=[checkpoint, board, log, term],
        )


if __name__ == "__main__":
    model = CaptchaModel()
    model.train()
