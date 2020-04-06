from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image
import numpy as np

w = 120
h = 40

img = img_to_array(Image.open(
    './images/captcha-001.png').convert("1")).reshape((h, w, 1))

print(img.dtype)

# img -= 40
# img /= 200
# img = np.around(img, 0)
# img -= 1.0

# print(img)

array_to_img(img).save('./debug-output.png')
