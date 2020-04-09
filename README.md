This is a CAPTCHA solver for [etimspayments](https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp).



### Approach:

Solving a captcha with a neural network is a multi-class problem, the network will receive as input a single character extracted from the captcha one at a time and then predicts for the given character what number it is (0-9). 

To do this the characters (0-9) need to be extracted from the captcha image. I found some existing code on how to do this from Adam Geitgey [here](https://s3-us-west-2.amazonaws.com/mlif-example-code/solving_captchas_code_examples.zip). I adapted the character extraction code he used to work with the [etimspayments](https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp) captcha images - it  works very well.

A simple convolutional network is all that is needed for this data, two convolutional layers, two max pooling layers, a hidden layer with 500 neurons and a final fully connected layer with ten output neurons for the ten classes. You may want to add more data but I just used 3000 downloaded captchas for training because it was fast and achieved a good accuracy. 



### Results:

The first model trained was using characters extracted from the fake generated captchas. This yielded a result of 87% accuracy on an unseen test set of 423 images, however it was not high enough. The low accuracy is probably due to the distribution of the fake data being substantially different to that of the real captchas from the website. 

The second model was trained on characters extracted from 500 real captcha images from [etimspayments](https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp). This resulted in a substantially better model that achieved 96% accuracy on the unseen test set of 423 real captcha images. This is, as just mentioned most likely due to the distribution of the training data being much closer to that of the unseen test images. This current model does not use fake generated captchas.

Update (April 9th 2020):

The third model has achieved 98.3% accuracy on the 423 unseen test images. This model was trained on extracted characters from 3000 downloaded captcha images from [etimspayments](https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp), the final number of characters it was trained on was ~10,000.



### Improvements:

At the moment 98.3% accuracy on 423 unseen test images is very good. 

Potential improvements to be made:

- Make sure the dataset is balanced with equal numbers of characters from each class.
- Tweak hyperparameters / network structure.



### Updates (April 9th 2020):

- Trained the network on a larger dataset of ~10,000 characters from 3000 captcha images.
- Created folder `/trainset_captchas` that the captcha images from [etimspayments](https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp) download to.
- Created folder `/trainset_chars` that contains the characters extracted from the downloaded captchas in `/trainset_captchas`.
- Renamed `/test_set` to `/testset_captchas` to maintain consistent naming scheme.
- Moved the data used to train the second model to 96% into a folder named `/96_trainset`.
- Updated `label_chars.py` by removing repeated, inefficient code. 
- Updated `extract_chars.py` to save all extracted chars to one folder - `/trainset_chars`, to be labeled later.
- Added `labeling_microsite` from [tmcw](https://github.com/tmcw)/**[howsmy](https://github.com/tmcw/howsmy)** to label captcha images.



### Usage:

Files:

- To extract characters from captcha images run `extract_chars.py`
- To generate fake captchas run `generate_captchas.py`
- To label new unlabeled training data using a separately trained model run `label_chars.py`
- `labels.dat` contains the data for converting to and from one-hot encodings using sklearn.
- To install required dependencies run `pip install -r requirements.txt`
- To test the trained model run `test.py`
- To train the convolutional network run `train.py`



Folders:

- `/96_trainset` contains the images and characters used for the 96% accuracy achieving model.
- `/chars` contains the characters used to generate fake captchas.
- `/models` contains the trained models.
- `/scripts` contains the shell script that downloads the captcha images.
- `/testset_captchas` contains the 423 captcha images used for testing the trained models.
- `/trainset_captchas` contains the 3000 captcha images that will be part of the training process.
- `/trainset_chars` contains the labeled characters that will be used to train the model, extracted from the `/trainset_captchas` images.



References:

- [Adam Geitgey](https://github.com/ageitgey) and his own [Captcha solver](https://s3-us-west-2.amazonaws.com/mlif-example-code/solving_captchas_code_examples.zip)
- https://stackoverflow.com/questions/59159257/cleaning-image-for-ocr/59166202#59166202