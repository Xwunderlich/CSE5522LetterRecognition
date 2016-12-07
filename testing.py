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
dataset.fold(10, 0)

neural_net = NNBuilder(dataset.attr_count, dataset.class_count)		# create a neural net with the number of input and the number of output
neural_net.add_layer(400, activation=tf.nn.relu)						# add a ReLU layer with certain amount of nodes
neural_net.add_layer(400, activation=tf.nn.relu)						# add a ReLU layer with certain amount of nodes
neural_net.add_layer(400, activation=tf.nn.relu)						# add a ReLU layer with certain amount of nodes
neural_net.finish()														# finish building the neural net
neural_net.train(dataset, rate=0.0000005, iteration=30000, peek_interval=100)
accuracy, loss = neural_net.evaluate(dataset)
time = format_time(neural_net.time)

print('RESULT: {:.2f}%   TIME: {}'.format(accuracy * 100, time))
