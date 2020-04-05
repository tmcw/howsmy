from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from PIL import Image
import numpy as np
import os
from keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import img_to_array
import toml

h = 40
w = 120

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
        print(opt)
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
        print('Loading train and validation data...')
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
