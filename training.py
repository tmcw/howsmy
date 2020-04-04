from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import toml

dataset = toml.load('./training.toml')

for (key, value) in dataset.items():
    file = open("./images/captcha-%s.png" % key.zfill(3), 'rb')
    img = img_to_array(Image.open(file).convert("L")).reshape((40, 120, 1))
    print(value)
