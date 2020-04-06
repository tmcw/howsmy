train:
	python train.py

tensorboard:
	tensorboard --logdir=logs

run:
	FLASK_APP=label.py python3 -m flask run
