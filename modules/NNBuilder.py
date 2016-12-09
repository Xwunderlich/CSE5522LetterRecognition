import tensorflow as tf
import time

class Var:
	@staticmethod
	def weights(*shape):
		init = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(init)

	@staticmethod
	def bias(*shape):
		init = tf.constant(0.1, shape=shape)
		return tf.Variable(init)
		
class Value:
	@staticmethod
	def sum(value, weight, bias):
		return tf.matmul(value, weight) + bias
		
	@staticmethod
	def cross_entropy(value, truth):
		loss = tf.nn.softmax_cross_entropy_with_logits(value, truth)
		return tf.reduce_mean(loss)
		
	@staticmethod
	def accuracy(value, truth):
		match = tf.equal(tf.argmax(value, 1), tf.argmax(truth, 1))
		return tf.reduce_mean(tf.cast(match, tf.float32))
		
	@staticmethod
	def conv2d(image, weight):
		return tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')

	@staticmethod
	def pool2x2(image):
		return tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
class NNBuilder:
	
	
	def __init__(self, features, classes):
		self.features = features
		self.classes = classes
		self.input = tf.placeholder(tf.float32, [None, features])
		self.labels = tf.placeholder(tf.float32, [None, classes])
		self.layers = [(self.input, features)]
		self.output = None
		self.loss = None
		self.accuracy = None
		self.train_step = None
		self.session = None
		self.start_time = None
		self.end_time = None
		
	@property
	def time(self):
		if self.end_time is None:
			return None
		return self.end_time - self.start_time
	
	def add_layer(self, nodes=None, activation=tf.nn.relu):
		if nodes is None:
			nodes = self.layers[-1][1]
		l, d = self.layers[-1]
		W = Var.weights(d, nodes)
		b = Var.bias(nodes)
		layer = activation(Value.sum(l, W, b))
		self.layers.append((layer, nodes))
		
	def finish(self):
		l, d = self.layers[-1]
		W = Var.weights(d, self.classes)
		b = Var.bias=(self.classes)
		self.output = Value.sum(l, W, b)
		self.loss = Value.cross_entropy(self.output, self.labels)
		self.accuracy = Value.accuracy(self.output, self.labels)
		
	def train(self, dataset, algorithm=tf.train.GradientDescentOptimizer, rate=0.1, iteration=10000, batch_size=100, peek_interval=1):
		self.session = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		self.train_step = algorithm(rate).minimize(self.loss)
		self.start_time = time.time()
		for i in range(iteration):
			batch_x, batch_y = dataset.train.next_batch(batch_size)
			self.session.run(self.train_step, feed_dict={self.input: batch_x, self.labels: batch_y})
			if i % peek_interval == 0:
				self.peek(i, dataset, (batch_x, batch_y))
		self.end_time = time.time()
				
	def peek(self, iteration, dataset, training_batch):
		test_accuracy, test_loss, train_accuracy, train_loss = self.evaluate(dataset, training_batch)
		str_iter = 'step: {}'.format(iteration)
		str_tr_acc = 'train accuracy: {:.2f}%'.format(train_accuracy * 100)
		str_tr_loss = 'train loss: {:.2f}'.format(train_loss)
		str_te_acc = 'test accuracy:  {:.2f}%'.format(test_accuracy * 100)
		str_te_loss = 'test loss:  {:.2f}'.format(test_loss)
		print('\n{}\n{:<25} {}\n{:<25} {}'.format(str_iter, str_tr_acc, str_tr_loss, str_te_acc, str_te_loss))
				
	def evaluate(self, dataset, training_batch=None):
		def eval(x, y):
			accuracy = self.session.run(self.accuracy, feed_dict={self.input: x, self.labels: y})
			loss = self.session.run(self.loss, feed_dict={self.input: x, self.labels: y})
			return accuracy, loss
		test_x, test_y = dataset.test.features, dataset.test.labels
		test_accuracy, test_loss = eval(test_x, test_y)
		if training_batch is not None:
			train_accuracy, train_loss = eval(*training_batch)
			return test_accuracy, test_loss, train_accuracy, train_loss
		return test_accuracy, test_loss
		
class CNNBuilder:
	def __init__(self):
		pass
		
