from PIL import Image, ImageDraw, ImageFont
import toml

dataset = list(toml.load('./training.toml').items())

for (num, label) in dataset:
    im = Image.open("./images/captcha-%s.png" % num.zfill(3)).resize((256, 256))
    canvas = Image.new('RGB', (512,256), (255,255,255))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    fnt = ImageFont.truetype('/Users/tmcw/Library/Fonts/Lunchtype22-Medium.ttf', 40)
    d.text((270,20), label, font=fnt, fill=(0,0,0))
    canvas.save('./synthetic/captcha-%s.png' % num.zfill(3))
