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
		
class Func:
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
	def image(value, height, width):
		return tf.reshape(value, [-1, height, width, 1])
		
	@staticmethod
	def flatten(image, height, width, depth):
		return tf.reshape(image, [-1, height * width * depth])
		
	@staticmethod
	def conv2d(image, weight, bias):
		return tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME') + bias

	@staticmethod
	def pool(image, height, width):
		return tf.nn.max_pool(image, ksize=[1, height, width, 1], strides=[1, height, width, 1], padding='SAME')
	
	
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
		self.keep_probs = {}
		self.test_probs = {}
		
	@property
	def time(self):
		if self.end_time is None:
			return None
		return self.end_time - self.start_time
	
	def add_dropout(self, layer, prob):
		if prob is None:
			return layer
		keep_prob = tf.placeholder(tf.float32)
		self.keep_probs[keep_prob] = prob
		self.test_probs[keep_prob] = 1.0
		return tf.nn.dropout(layer, keep_prob)
	
	def add_layer(self, nodes=None, activation=tf.nn.relu, dropout=None):
		l, d = self.layers[-1]
		if nodes is None:
			nodes = d
		l = self.add_dropout(l, dropout)
		W = Var.weights(d, nodes)
		b = Var.bias(nodes)
		layer = activation(Func.sum(l, W, b))
		self.layers.append((layer, nodes))
		
	def finish(self, dropout=None):
		l, d = self.layers[-1]
		l = self.add_dropout(l, dropout)
		W = Var.weights(d, self.classes)
		b = Var.bias=(self.classes)
		self.output = Func.sum(l, W, b)
		self.loss = Func.cross_entropy(self.output, self.labels)
		self.accuracy = Func.accuracy(self.output, self.labels)
		
	def train(self, dataset, algorithm=tf.train.GradientDescentOptimizer, rate=0.1, iteration=10000, batch_size=100, peek_interval=10, show_test=True):
		self.session = tf.InteractiveSession()
		self.train_step = algorithm(rate).minimize(self.loss)
		self.start_time = time.time()
		tf.global_variables_initializer().run()
		for i in range(iteration):
			batch_x, batch_y = dataset.train.next_batch(batch_size)
			self.session.run(self.train_step, feed_dict={self.input: batch_x, self.labels: batch_y, **self.keep_probs})
			if i % peek_interval == 0:
				self.peek(i, dataset if show_test else None, (batch_x, batch_y))
		self.end_time = time.time()
				
	def peek(self, iteration, dataset, training_batch):
		str_iter = 'step: {}'.format(iteration)
		print('\n{}'.format(str_iter))
		if training_batch is not None:
			train_accuracy, train_loss = self.evaluate(batch=training_batch)
			str_tr_acc = 'train accuracy: {:.2f}%'.format(train_accuracy * 100)
			str_tr_loss = 'train loss: {:.2f}'.format(train_loss)
			print('{:<25} {}'.format(str_tr_acc, str_tr_loss))
		if dataset is not None:
			test_accuracy, test_loss = self.evaluate(dataset=dataset)
			str_te_acc = 'test accuracy:  {:.2f}%'.format(test_accuracy * 100)
			str_te_loss = 'test loss:  {:.2f}'.format(test_loss)
			print('{:<25} {}'.format(str_te_acc, str_te_loss))
				
	def evaluate(self, dataset=None, batch=None):
		def eval(x, y):
			accuracy = self.session.run(self.accuracy, feed_dict={self.input: x, self.labels: y, **self.test_probs})
			loss = self.session.run(self.loss, feed_dict={self.input: x, self.labels: y, **self.test_probs})
			return accuracy, loss
		if dataset is not None:
			test_x, test_y = dataset.test.features, dataset.test.labels
			test_accuracy, test_loss = eval(test_x, test_y)
			return test_accuracy, test_loss
		if batch is not None:
			batch_accuracy, batch_loss = eval(*batch)
			return batch_accuracy, batch_loss
		return None, None
		
		
class CNNBuilder(NNBuilder):
	def __init__(self, height, width, classes):
		super().__init__(width * height, classes)
		self.width = width
		self.height = height
		
	def add_linker(self, type):
		last_type = len(self.layers[-1])
		if type == 'convolve' and last_type == 2:
			image = Func.image(self.input, self.height, self.width)
			self.layers.append((image, self.height, self.width, 1))
		if type == 'flat' and last_type == 4:
			img, h, w, d = self.layers[-1]
			layer = Func.flatten(img, h, w, d)
			self.layers.append((layer, h * w * d))
		
	def add_conv_layer(self, depth=None, filter_size=(5, 5), activation=tf.nn.relu):
		self.add_linker('convolve')
		l, h, w, d = self.layers[-1]
		if depth is None:
			depth = d
		W = Var.weights(*filter_size, d, depth)
		b = Var.bias(depth)
		layer = activation(Func.conv2d(l, W, b))
		self.layers.append((layer, h, w, depth))
		
	def add_pool_layer(self, height=2, width=2):
		l, h, w, d = self.layers[-1]
		layer = Func.pool(l, height, width)
		new_height = h // height
		new_width = w // width
		self.layers.append((layer, new_height, new_width, d))
		
	def add_layer(self, nodes=None, activation=tf.nn.relu, dropout=None):
		self.add_linker('flat')
		super().add_layer(nodes, activation, dropout)
		
	def finish(self, dropout=None):
		self.add_linker('flat')
		super().finish(dropout)		
