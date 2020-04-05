from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import toml

# Script that runs the trained model against captchas to showcase whether
# we're starting to actually recognize them.

w = 120
h = 40
model_path = "./checkpoints/model.hdf5"
alphabet = "0123456789"
char_size = len(alphabet)
categories = 5

def load_image(path):
    img = img_to_array(Image.open(path).convert("L")).reshape((h, w, 1))
    img /= 127.5
    img -= 1.0
    return img

def vec2text(label):
    arr = [l.argmax().tolist() for l in label.reshape((categories, char_size))]
    ret = [alphabet[l] for l in arr]
    return ''.join(ret)

model = load_model(model_path)

dataset = list(toml.load('./training.toml').items())

for (num, label) in dataset:
    image = load_image('./images/captcha-%s.png' % num.zfill(3))
    predicted = model.predict(image.reshape((1, h, w, 1)))
    print('captcha-%s: %s ≈ %s' % (num.zfill(3), label, vec2text(predicted)))
