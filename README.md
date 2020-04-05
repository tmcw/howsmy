# howsmy

Wow! Okay!

This repo is the bones of a solver for the CAPTCHA on [etimspayments](https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp),
which is the key to unlocking [howsmydrivingny](https://twitter.com/howsmydrivingny) but for
San Francisco. This is my first adventure in 'Machine Learning' so it's really chaotic and
crazy. Some notes below:

- `train.py` is the training script, the only relevant Python script. It uses PIL, Tensorflow,
  Keras, and other tools to try and replicate the gist of [YACS](https://github.com/yeguixin/captcha_solver).
- `make_char_mashups.py` generates 100,000 fake captchas in the `generated` folder
- `training.toml` contains 557 trained labels in TOML format, (filename = label).
- `images` contains the images labeled by those labels.
- `chars` contains sliced-up numbers used to generate fake captchas

To run:

You'll need PIL (or Pillow) and Tensorflow

Make generated data:

```
mkdir generated
python make_char_mashups.py
```

Run the model:

```
python train.py
```

---


Current status is **not great**: I have been embarrassingly plugging in different
potential 'fixes' and everything either yields wildly expanding loss (3750857.9531),
or accuracy pinned at 1, or nans.

Current tweaks implemented:

1. Images are converted to black & white immediately
2. Bottom two layers of the model are removed. I _think_ bottom means 'larger, and lower in the code, and later in the flow',
   but it could also mean the opposite.
3. Removing `activation=softmax` dramatically changes the model: the loss becomes less
   absurd, but also the accuracy doesn't increase.

This being a blend of [Jeff's code](https://gist.github.com/thejefflarson/d8e2a65f37a37d39309058d23f6a71f1) and
[YACS](https://github.com/yeguixin/captcha_solver), there are a bunch of differences between those two
codebases and I'm not sure which is better in this case:

In format thejefflarson vs yeguixin:

- RMSprop or adam for the optimizer?
- learning_rate of 0.0001, or 0.01?
