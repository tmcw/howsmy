# Originally named: make_char_mashups.py

from PIL import Image, ImageDraw, ImageFont
from os import listdir
from random import randint
import toml

char_files = listdir('./chars');

chars = {
    '0': [],
    '1': [],
    '2': [],
    '3': [],
    '4': [],
    '5': [],
    '6': [],
    '7': [],
    '8': [],
    '9': []
}

for f in char_files:
    num = f[0]
    chars[num].append(Image.open('./chars/%s' % f))


def random_captcha():
    return ''.join(map(lambda x: str(x), [
        randint(0, 9),
        randint(0, 9),
        randint(0, 9),
        randint(0, 9),
        randint(0, 9)
    ]))


def get_img(i):
    options = chars[i]
    return options[randint(0, len(options) - 1)]


i = 0
for i in range(0, 100000):
    canvas = Image.new('RGB', (120,40), (255,255,255))

    cap = random_captcha()
    x = 0

    for c in cap:
        im = get_img(c)
        canvas.paste(im, (x, 0))
        x = x + im.width

    canvas.save('./generated_captchas/%s.png' % (cap))
