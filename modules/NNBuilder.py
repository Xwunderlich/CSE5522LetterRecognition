import tensorflow as tf
import time

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
		W = tf.Variable(tf.truncated_normal([d, nodes]))
		b = tf.Variable(tf.zeros([nodes]))
		layer = activation(tf.matmul(l, W) + b)
		self.layers.append((layer, nodes))
		
	def finish(self):
		l, d = self.layers[-1]
		W = tf.Variable(tf.truncated_normal([d, self.classes]))
		b = tf.Variable(tf.zeros([self.classes]))
		self.output = tf.matmul(l, W) + b
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels))
		prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.labels, 1))
		self.accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
		
	def train(self, dataset, rate=0.1, iteration=10000, batch_size=100, peek_interval=1):
		self.session = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		self.train_step = tf.train.GradientDescentOptimizer(rate).minimize(self.loss)
		self.start_time = time.time()
		for i in range(iteration):
			batch_x, batch_y = dataset.train.next_batch(batch_size)
			self.session.run(self.train_step, feed_dict={self.input: batch_x, self.labels: batch_y})
			if i % peek_interval == 0:
				test_accuracy, test_loss, train_accuracy, train_loss = self.evaluate(dataset, (batch_x, batch_y))
				str_iter = 'iteration: {}'.format(i)
				str_tr_acc = 'train accuracy: {:.2f}%'.format(train_accuracy * 100)
				str_tr_loss = 'train loss: {:.2f}'.format(train_loss)
				str_te_acc = 'test accuracy:  {:.2f}%'.format(test_accuracy * 100)
				str_te_loss = 'test loss:  {:.2f}'.format(test_loss)
				print('{}\n{:<25} {}\n{:<25} {}\n'.format(str_iter, str_tr_acc, str_tr_loss, str_te_acc, str_te_loss))
		self.end_time = time.time()
				
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
		
