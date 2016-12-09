import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from modules.DatasetLoader import *
from modules.NNBuilder import *

def format_time(time):
	hour = int(time / 3600)
	minute = int((time % 3600) / 60)
	second = time % 60
	return '{}:{:02}:{:05.2f}'.format(hour, minute, second)

dataset = DatasetLoader('MNIST_data/statistical.data', class_list=DatasetLoader.digits)
dataset.load(one_hot=True)
dataset.fold(10, test_fold=9)

neural_net = CNNBuilder(dataset.attr_count, dataset.class_count)		# create a neural net with the number of input and the number of output
neural_net.add_layer(500, activation=tf.nn.relu)						# add a ReLU layer with certain amount of nodes
neural_net.add_layer(500, activation=tf.nn.relu)						# add a ReLU layer with certain amount of nodes
neural_net.add_layer(500, activation=tf.nn.relu)						# add a ReLU layer with certain amount of nodes
neural_net.finish(dropout=0.5)														# finish building the neural net
neural_net.train(dataset, algorithm=tf.train.AdamOptimizer, rate=1e-4, iteration=200000, batch_size=100, peek_interval=100)
accuracy, loss, = neural_net.evaluate(dataset)
time = format_time(neural_net.time)

print('RESULT: {:.2f}%   TIME: {}'.format(accuracy * 100, time))