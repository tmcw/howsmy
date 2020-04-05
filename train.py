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
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from PIL import Image
import numpy as np
import os
from keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import img_to_array
import toml


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

def load_image(path):
    img = img_to_array(Image.open(path).convert("L")).reshape((h, w, 1))
    img /= 127.5
    img -= 1.0
    return img

def char2dict(char_set):
    dict = {}
    for i, char in enumerate(char_set):
        dict[i] = char
    return dict


def create_dict(cap_scheme):
    scheme = CaptchaSchemes()
    return char2dict(scheme.ebay)

class CaptchaSchemes:
    def __init__(self):
        self.ebay = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]


class Generator:
    def __init__(self, opt):
        # print(opt)
        self.opt = opt
        self.datagen = image.ImageDataGenerator(
            preprocessing_function=preprocess_input,
            # rotation_range=15,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # shear_range=0.2,
            # zoom_range=0.2,
            # horizontal_flip=True
        )
        self.dict = create_dict('ebay')
        # print('Loading train and validation data...')
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.data_load()

    def text2vec(self, text):
        text_len = len(text)
        if text_len > self.opt.cap_len:
            raise ValueError('The max length of this captcha is {}' .format(self.opt.cap_len))
        if self.opt.char_set_len != len(self.dict):
            raise ValueError('The number of characters does not match to the dict')
        vector = np.zeros(self.opt.cap_len * self.opt.char_set_len)

        def char2pos(c):
            k = -1
            for (key, value) in self.dict.items():
                if value == c:
                    k = key
                    return k
            if k == -1:
                raise ValueError('Wrong with dict or text')
        for i, c in enumerate(text):
            idx = i * self.opt.char_set_len + char2pos(c)
            vector[idx] = 1
        return vector

    # load training and validation data
    def data_load(self):
        dataset = list(toml.load('./training.toml').items())

        training_slice = dataset[0:400]
        testing_slice = dataset[400:]

        self.num_train_samples =min(self.opt.train_size, len(training_slice))
        self.num_test_sample = min(2000, len(testing_slice))

        # load training set
        x_train = np.empty((self.num_train_samples, self.opt.loadHeight, self.opt.loadWidth, 1), dtype='uint8')
        y_train = np.empty((self.num_train_samples, self.opt.cap_len * self.opt.char_set_len), dtype='uint8')
        for i, (num, label) in enumerate(training_slice):
            img_name = os.path.join('./images/', 'captcha-%s.png' % num.zfill(3))
            x_train[i, :, :, :] = load_image(img_name)
            y_train[i, :] = self.text2vec(label)

        # load testing set
        x_test = np.empty((self.num_test_sample, self.opt.loadHeight, self.opt.loadWidth, 1), dtype='uint8')
        y_test = np.empty((self.num_test_sample, self.opt.cap_len * self.opt.char_set_len), dtype='uint8')
        for i, (num, label) in enumerate(testing_slice):
            img_name = os.path.join('./images/', 'captcha-%s.png' % num.zfill(3))
            x_train[i, :, :, :] = load_image(img_name)
            y_train[i, :] = self.text2vec(label)

        return (x_train, y_train), (x_test, y_test)

    # Synthetic data generator
    def synth_generator(self, phase):
        if phase == 'train':
            return self.datagen.flow(self.x_train, self.y_train, batch_size=self.opt.batchSize)
        elif phase == 'val':
            return self.datagen.flow(self.x_test, self.y_test, batch_size=self.opt.batchSize, shuffle=False)
        else:
            raise ValueError('Please input train or val phase')

generator = Generator(Opt())

# def reshape_generator(gen):
#     for (x, y) in gen:
#         # print(y, y.shape)
#         y = np.array(y).reshape((32, -1))
#         yield (x, y)

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
