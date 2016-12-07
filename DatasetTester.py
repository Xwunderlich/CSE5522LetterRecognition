import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from DatasetLoader import *
from NNBuilder import *

def format_time(time):
	hour = int(time / 3600)
	minute = int((time % 3600) / 60)
	second = time % 60
	return '{}:{:02}:{:05.2f}'.format(hour, minute, second)

dataset = DatasetLoader('MNIST_statistical.data', class_list=list(str(i) for i in range(10)))
dataset.load(one_hot=True)
dataset.fold(10, 0)

neural_net = NNBuilder(dataset.attr_count, dataset.class_count)
neural_net.add_layer(196, activation=tf.nn.relu)
neural_net.add_layer(196, activation=tf.nn.relu)
neural_net.add_layer(196, activation=tf.nn.relu)
neural_net.finish()
neural_net.train(dataset, rate=0.0000001, iteration=30000, report_interval=100)
accuracy, loss = neural_net.evaluate(dataset)
time = format_time(neural_net.time)

print('RESULT: {:.2f}%   TIME: {}'.format(accuracy * 100, time))
