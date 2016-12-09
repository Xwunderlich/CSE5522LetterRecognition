import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from modules.DatasetLoader import *
from modules.NNBuilder import *

def format_time(time):
	hour = int(time / 3600)
	minute = int((time % 3600) / 60)
	second = time % 60
	return '{}:{:02}:{:05.2f}'.format(hour, minute, second)

dataset = DatasetLoader('Stanford_data/raw.data', class_list=DatasetLoader.lower_letters)
dataset.load(one_hot=True)
dataset.fold(10, test_fold=9)

neural_net = CNNBuilder(16, 8, dataset.class_count)						# create a convolutional neural net
neural_net.add_conv_layer(32, filter_size=(5, 5), activation=tf.nn.relu)	# add a colvolution layer with certain depth and filter size
neural_net.add_pool_layer(2, 2)												# add a 2x2 max pool layer
neural_net.add_conv_layer(64, filter_size=(5, 5), activation=tf.nn.relu)	# add a colvolution layer with certain depth and filter size
neural_net.add_pool_layer(2, 2)												# add a 2x2 max pool layer
neural_net.add_layer(1024, activation=tf.nn.relu)							# add a ReLU layer with certain amount of nodes
neural_net.finish(dropout=0.5)												# finish building the neural net with a dropout on the output layer
neural_net.train(dataset, algorithm=tf.train.AdamOptimizer, rate=2e-5, iteration=200000, batch_size=50, peek_interval=100)
accuracy, loss, = neural_net.evaluate(dataset)
time = format_time(neural_net.time)

print('RESULT: {:.2f}%   TIME: {}'.format(accuracy * 100, time))