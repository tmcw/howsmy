# howsmy

Wow! Okay!

This repo is the bones of a solver for the CAPTCHA on [etimspayments](https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp),
which is the key to unlocking [howsmydrivingny](https://twitter.com/howsmydrivingny) but for
San Francisco. This is my first adventure in 'Machine Learning' so it's really chaotic and
crazy. Some notes below:

- `train.py` is the training script, the only relevant Python script. It uses PIL, Tensorflow,
  Keras, and other tools to try and replicate the gist of [YACS](https://github.com/yeguixin/captcha_solver).
- `training.toml` contains 557 trained labels in TOML format, (filename = label).
- `images` contains the images labeled by those labels.

---

Current status:

Bashed through the initial crashing errors and misunderstandings and got to the point where:

1. It runs,
2. But it isn't learning
3. And it complains about 'running out of data'.

So there are basically two approaches to training data: generating training data by
generating CAPTCHAS, and multiplying training data using pix2pix (keras 'inception' in this case).
This does the _latter_ and not the former, yet. Doing the former is high on my list, though
the whole 'having a lot of similar fonts' step is not quite clear: how similar do these
generated captchas have to be to your target?

Current example output from training:

```
➜  howsmy git:(master) ✗ python3 train.py
Using TensorFlow backend.
2020-04-04 17:46:47.659877: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-04-04 17:46:47.669602: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f92f9325070 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-04-04 17:46:47.669613: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From train.py:285: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
Instructions for updating:
Please use Model.fit, which supports generators.
WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to
  ['...']
Train for 20 steps
Epoch 1/1000
2020-04-04 17:46:48.633299: I tensorflow/core/profiler/lib/profiler_session.cc:225] Profiler session started.
 4/20 [=====>........................] - ETA: 9s - loss: 22.4280 - accuracy: 0.1100 2020-04-04 17:46:50.349267: W tensorflow/core/common_runtime/base_collective_executor.cc:217] BaseCollectiveExecutor::StartAbort Out of range: End of sequence
	 [[{{node IteratorGetNext}}]]
WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 20000 batches). You may need to use the repeat() function when building your dataset.
7b0b4 [0.02181366 0.01992532 0.02248489 0.01303369 0.02129689] ['9', '7', '0', '7', '7'] ['9', '0', '7', '8', '0']
97f69 [0.0218948  0.01915686 0.02228328 0.01292676 0.02113249] ['9', '7', '0', '7', '7'] ['8', '2', '4', '8', '1']
b5acc [0.02167206 0.01937111 0.02212382 0.01338832 0.02113979] ['9', '7', '0', '7', '7'] ['0', '0', '8', '6', '3']
d6d81 [0.02189644 0.01973893 0.02219808 0.01336587 0.02109278] ['9', '7', '0', '7', '7'] ['3', '2', '6', '6', '5']
30dba [0.02220587 0.01944711 0.02252331 0.01312182 0.02109946] ['9', '7', '0', '7', '7'] ['9', '4', '0', '0', '0']
 4/20 [=====>........................] - ETA: 10s - loss: 22.4719 - accuracy: 0.1100Epoch 2/1000
 4/20 [=====>........................] - ETA: 7s - loss: 35.9904 - accuracy: 0.13002020-04-04 17:46:52.465497: W tensorflow/core/common_runtime/base_collective_executor.cc:217] BaseCollectiveExecutor::StartAbort Out of range: End of sequence
	 [[{{node IteratorGetNext}}]]
WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 20000 batches). You may need to use the repeat() function when building your dataset.
a498b [0.02506818 0.01982114 0.02153236 0.00269465 0.01811286] ['9', '7', '0', '7', '7'] ['0', '0', '0', '0', '0']
dfe75 [0.02525185 0.01969707 0.02128096 0.00267149 0.01798204] ['9', '7', '0', '7', '7'] ['0', '0', '0', '0', '0']
33154 [0.02486924 0.01979077 0.02160252 0.00270216 0.018137  ] ['6', '7', '0', '7', '7'] ['0', '0', '0', '0', '0']
3a5be [0.02484172 0.01954813 0.02141079 0.00271831 0.01813234] ['9', '7', '0', '7', '7'] ['0', '0', '0', '0', '0']
08363 [0.02520425 0.01983635 0.0215323  0.00268751 0.01824794] ['9', '7', '0', '7', '7'] ['0', '0', '0', '0', '0']
 4/20 [=====>........................] - ETA: 8s - loss: 36.6354 - accuracy: 0.1300Epoch 3/1000
 1/20 [>.............................] - ETA: 11sWARNING:tensorflow:Can save best model only with accuracy available, skipping.
```

This project also doesn't use virtualenv or anything particularly mature about Python
packaging. That's a TODO.

---

Main questions on my mind:

- Is generating new, local captchas the key to moving past the current blocker?
- As far as I can tell, the keras inception thing is an _infinite_ generator. How is
  Tensorflow running out of data?
