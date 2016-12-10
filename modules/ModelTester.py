import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from modules.DatasetLoader import *
from modules.NNBuilder import *

class ModelTester:
	def __init__(self):
		self.dataset = None
		self.model = None
		
	@staticmethod
	def format_time(time):
		hour = int(time / 3600)
		minute = int((time % 3600) / 60)
		second = time % 60
		return '{}:{:02}:{:05.2f}'.format(hour, minute, second)
		
	def prepare_data(self, dataset_path, class_list, fold, test_fold):
		self.dataset = DatasetLoader(dataset_path, class_list)
		self.dataset.load(one_hot=True)
		self.dataset.fold(fold, test_fold)
		
	def build_NN_model(self, layers=[], activation=tf.nn.relu, dropout=None):
		self.model = NNBuilder(self.dataset.attr_count, self.dataset.class_count)
		for layer in layers:
			self.model.add_layer(layer, activation=activation)
		self.model.finish(dropout=dropout)
		
	def build_CNN_model(self, height, width, conv_layers=[], dense_layers=[], activation=tf.nn.relu, dropout=None):
		self.model = CNNBuilder(height, width, dataset.class_count)
		for conv in conv_layers:
			self.model.add_conv_layer(conv, filter_size=(5, 5), activation=activation)	
			self.model.add_pool_layer(2, 2)	
		for dense in dense_layers:
			self.model.add_layer(dense, activation=activation)	
		self.model.finish(dropout=dropout)
		
	def train_model(self, algorithm, rate, iteration, batch_size=100, peek_interval=100, show_test=True):
		self.model.train(self.dataset, algorithm=algorithm, rate=rate, iteration=iteration, batch_size=batch_size, peek_interval=peek_interval, show_test=show_test)
		
	def evaluate_model(self):
		accuracy, loss = self.model.evaluate(dataset=self.dataset)
		time = self.format_time(self.model.time)
		print('RESULT: {:.2f}%   TIME: {}'.format(accuracy * 100, time))