from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image, ImageEnhance
import numpy as np

w = 100
h = 20

im = ImageEnhance.Brightness(Image.open('./images/captcha-001.png').crop((0, 0, w, 20))).enhance(1.5).convert("1")
img = img_to_array(im).reshape((h, w, 1))

array_to_img(img).save('./debug-output.png')
