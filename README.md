This is a CAPTCHA solver for [etimspayments](https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp).



#### Approach:

Solving a captcha with a neural network is a multi-class problem, the network will receive inputs of characters from the captcha one at a time and then predict for the given character what number it is (0-9). 

To do this the characters (0-9) need to be extracted from the captcha image. I found some existing code on how to do this from Adam Geitgey [here](https://s3-us-west-2.amazonaws.com/mlif-example-code/solving_captchas_code_examples.zip). I adapted the character extraction code he used to work with the [etimspayments](https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp) captcha images - it  works very well.

A simple convolutional network is all that is needed for this data, two convolutional layers, two max pooling layers, a hidden layer with 500 neurons and a final fully connected layer with ten output neurons for the ten classes. You may want to add more data but I just used about 5000 examples for training because it was fast and the characters aren't that complex. 



#### Results:

The first model trained was using characters extracted from the fake generated captchas. This yielded a result of 83% accuracy on an unseen test set, however it was not high enough. The low accuracy is probably due to the distribution of the fake data being substantially different to that of the real captchas from the website. 

The second model was trained on characters extracted from the real captcha images from [etimspayments](https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp). This resulted in a substantially better model that achieved 96% accuracy on an unseen test set of 423 real captcha images. This is, as just mentioned most likely due to the distribution of the training data being much closer to that of the unseen test images. This current model does not use fake generated captchas.



#### Improvements:

At the moment 96% accuracy on 423 unseen test images is very promising. However, with more training data, tweaking of hyperparameters and image pre-processing the true accuracy of the model is likely higher.



#### Usage:

Files:

- To extract labeled characters from the generated captchas run ```extract_chars.py```
- To generate fake captchas run ```generate_captchas.py``` 
- To train the convolutional network run ```train.py``` 
- To test the trained model run ```test.py```
- To label new unlabeled training data using a separately trained model run ```label_chars.py```
- ```labels.dat``` contains the data for converting to and from one-hot encodings using sklearn

Folders:

- ```/chars``` contains the characters used to generate fake captchas
- ```/models``` contains the trained models
- ```/extracted_chars_real``` is the output folder for the extracted chars from real captchas
- ```/extracted_chars_fake``` is the output folder for the extracted chars from fake generated captchas
- ```/generated_captchas``` contains the generated captchas from ```generate_captchas.py```
- ```/images``` contains the real captchas used for extraction and training, downloaded from [etimspayments](https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp) 
- ```/test_set``` contains a test set of real captchas downloaded from  [etimspayments](https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp) using ```download.sh```





References:

- [Adam Geitgey](https://github.com/ageitgey) and his own [Captcha solver](https://s3-us-west-2.amazonaws.com/mlif-example-code/solving_captchas_code_examples.zip)
- https://stackoverflow.com/questions/59159257/cleaning-image-for-ocr/59166202#59166202