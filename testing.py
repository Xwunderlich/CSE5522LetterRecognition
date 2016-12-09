import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from modules.DatasetLoader import *
from modules.NNBuilder import *

def format_time(time):
	hour = int(time / 3600)
	minute = int((time % 3600) / 60)
	second = time % 60
	return '{}:{:02}:{:05.2f}'.format(hour, minute, second)

def test_MNIST_raw_CNN():
	dataset = DatasetLoader('MNIST_data/raw', class_list=DatasetLoader.digits)
	dataset.load(one_hot=True)
	dataset.fold(10, test_fold=9)

	neural_net = CNNBuilder(28, 28, dataset.class_count)						# create a convolutional neural net
	neural_net.add_conv_layer(32, filter_size=(5, 5), activation=tf.nn.relu)	# add a colvolution layer with certain depth and filter size
	neural_net.add_pool_layer(2, 2)												# add a 2x2 max pool layer
	neural_net.add_conv_layer(64, filter_size=(5, 5), activation=tf.nn.relu)	# add a colvolution layer with certain depth and filter size
	neural_net.add_pool_layer(2, 2)												# add a 2x2 max pool layer
	neural_net.add_layer(1024, activation=tf.nn.relu)							# add a densely connected ReLU layer with certain amount of nodes
	neural_net.finish(dropout=0.5)												# finish building the neural net with a dropout on the output layer
	neural_net.train(dataset, algorithm=tf.train.AdamOptimizer, rate=2e-5, iteration=20000, batch_size=50, peek_interval=10, show_test=False)
	accuracy, loss = neural_net.evaluate(dataset)
	time = format_time(neural_net.time)

	print('RESULT: {:.2f}%   TIME: {}'.format(accuracy * 100, time))

def test_MNIST_statistical():
	dataset = DatasetLoader('MNIST_data/statistical', class_list=DatasetLoader.digits)
	dataset.load(one_hot=True)
	dataset.fold(10, test_fold=9)

	neural_net = NNBuilder(dataset.attr_count, dataset.class_count)		# create a neural net with the number of input and the number of output
	neural_net.add_layer(500, activation=tf.nn.relu)						# add a ReLU layer with certain amount of nodes
	neural_net.add_layer(500, activation=tf.nn.relu)						# add a ReLU layer with certain amount of nodes
	neural_net.add_layer(500, activation=tf.nn.relu)						# add a ReLU layer with certain amount of nodes
	neural_net.finish(dropout=0.5)											# create the output layer with a dropout
	neural_net.train(dataset, algorithm=tf.train.AdamOptimizer, rate=1e-4, iteration=200000, batch_size=100, peek_interval=100)
	accuracy, loss = neural_net.evaluate(dataset)
	time = format_time(neural_net.time)
	print('RESULT: {:.2f}%   TIME: {}'.format(accuracy * 100, time))
	
def test_MNIST_diagonal():
	dataset = DatasetLoader('MNIST_data/diagonal', class_list=DatasetLoader.digits)
	dataset.load(one_hot=True)
	dataset.fold(10, test_fold=9)

	neural_net = NNBuilder(dataset.attr_count, dataset.class_count)		# create a neural net with the number of input and the number of output
	neural_net.add_layer(500, activation=tf.nn.relu)						# add a ReLU layer with certain amount of nodes
	neural_net.add_layer(500, activation=tf.nn.relu)						# add a ReLU layer with certain amount of nodes
	neural_net.add_layer(500, activation=tf.nn.relu)						# add a ReLU layer with certain amount of nodes
	neural_net.finish(dropout=0.5)											# create the output layer with a dropout
	neural_net.train(dataset, algorithm=tf.train.AdamOptimizer, rate=1e-4, iteration=200000, batch_size=100, peek_interval=100)
	accuracy, loss = neural_net.evaluate(dataset)
	time = format_time(neural_net.time)
	print('RESULT: {:.2f}%   TIME: {}'.format(accuracy * 100, time))

def test_stanford_raw_softmax():
	dataset = DatasetLoader('Stanford_data/raw', class_list=DatasetLoader.lower_letters)
	dataset.load(one_hot=True)
	dataset.fold(10, test_fold=9)

	neural_net = NNBuilder(dataset.attr_count, dataset.class_count)
	neural_net.finish()												
	neural_net.train(dataset, algorithm=tf.train.AdamOptimizer, rate=1e-3, iteration=30000, batch_size=1000, peek_interval=100)
	accuracy, loss = neural_net.evaluate(dataset)
	time = format_time(neural_net.time)

	print('RESULT: {:.2f}%   TIME: {}'.format(accuracy * 100, time))

def test_stanford_raw_CNN():
	dataset = DatasetLoader('Stanford_data/raw', class_list=DatasetLoader.lower_letters)
	dataset.load(one_hot=True)
	dataset.fold(10, test_fold=9)

	neural_net = CNNBuilder(16, 8, dataset.class_count)						# create a convolutional neural net
	neural_net.add_conv_layer(32, filter_size=(5, 5), activation=tf.nn.relu)	# add a colvolution layer with certain depth and filter size
	neural_net.add_pool_layer(2, 2)												# add a 2x2 max pool layer
	neural_net.add_conv_layer(64, filter_size=(5, 5), activation=tf.nn.relu)	# add a colvolution layer with certain depth and filter size
	neural_net.add_pool_layer(2, 2)												# add a 2x2 max pool layer
	neural_net.add_layer(1024, activation=tf.nn.relu)							# add a densely connected ReLU layer with certain amount of nodes
	neural_net.finish(dropout=0.5)												# finish building the neural net with a dropout on the output layer
	neural_net.train(dataset, algorithm=tf.train.AdamOptimizer, rate=2e-5, iteration=200000, batch_size=50, peek_interval=500)
	accuracy, loss = neural_net.evaluate(dataset)
	time = format_time(neural_net.time)

	print('RESULT: {:.2f}%   TIME: {}'.format(accuracy * 100, time))
	
test_stanford_raw_softmax()