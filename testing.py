import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from modules.DatasetLoader import *
from modules.NNBuilder import *
from modules.ModelTester import *


def test_MNIST_raw_softmax():
	tester = ModelTester()
	tester.prepare_data('MNIST_data/raw', class_list=DatasetLoader.digits, fold=10, test_fold=0)
	tester.build_NN_model([], activation=tf.nn.relu, dropout=None)											
	tester.train_model(algorithm=tf.train.GradientDescentOptimizer, rate=2e-3, iteration=200000, batch_size=200, peek_interval=1000)
	tester.evaluate_model()

def test_MNIST_raw_1_layers():
	tester = ModelTester()
	tester.prepare_data('MNIST_data/raw', class_list=DatasetLoader.digits, fold=10, test_fold=0)
	tester.build_NN_model([1000], activation=tf.nn.relu, dropout=None)											
	tester.train_model(algorithm=tf.train.AdamOptimizer, rate=1e-3, iteration=100000, batch_size=100, peek_interval=500)
	tester.evaluate_model()
		
def test_MNIST_raw_3_layers():
	tester = ModelTester()
	tester.prepare_data('MNIST_data/raw', class_list=DatasetLoader.digits, fold=10, test_fold=0)
	tester.build_NN_model([500, 500, 500], activation=tf.nn.relu, dropout=0.5)											
	tester.train_model(algorithm=tf.train.AdamOptimizer, rate=5e-5, iteration=20000, batch_size=100, peek_interval=100)
	tester.evaluate_model()

def test_MNIST_raw_CNN():
	tester = ModelTester()
	tester.prepare_data('MNIST_data/raw', class_list=DatasetLoader.digits, fold=10, test_fold=0)
	tester.build_CNN_model(28, 28, conv_layers=[32, 64], dense_layers=[1024], activation=tf.nn.relu, dropout=0.5)
	tester.train_model(algorithm=tf.train.AdamOptimizer, rate=1e-4, iteration=20000, batch_size=50, peek_interval=10, show_test=False)
	tester.evaluate_model()

def test_MNIST_statistical():
	tester = ModelTester()
	tester.prepare_data('MNIST_data/statistical', class_list=DatasetLoader.digits, fold=10, test_fold=0)
	tester.build_NN_model([500, 500, 500], activation=tf.nn.relu, dropout=0.5)											
	tester.train_model(algorithm=tf.train.AdamOptimizer, rate=1e-4, iteration=200000, batch_size=100, peek_interval=100)
	tester.evaluate_model()
	
def test_MNIST_diagonal():
	tester = ModelTester()
	tester.prepare_data('MNIST_data/diagonal', class_list=DatasetLoader.digits, fold=10, test_fold=0)
	tester.build_NN_model([500, 500, 500], activation=tf.nn.relu, dropout=0.5)											
	tester.train_model(algorithm=tf.train.AdamOptimizer, rate=1e-4, iteration=200000, batch_size=100, peek_interval=100)
	tester.evaluate_model()
	
def test_MNIST_centroid():
	tester = ModelTester()
	tester.prepare_data('MNIST_data/centroid', class_list=DatasetLoader.digits, fold=10, test_fold=0)
	tester.build_NN_model([500, 500, 500], activation=tf.nn.relu, dropout=0.5)											
	tester.train_model(algorithm=tf.train.AdamOptimizer, rate=1e-4, iteration=200000, batch_size=100, peek_interval=100)
	tester.evaluate_model()
	
def test_MNIST_hotspot():
	tester = ModelTester()
	tester.prepare_data('MNIST_data/hotspot', class_list=DatasetLoader.digits, fold=10, test_fold=0)
	tester.build_NN_model([500, 500, 500], activation=tf.nn.relu, dropout=0.5)											
	tester.train_model(algorithm=tf.train.AdamOptimizer, rate=1e-4, iteration=200000, batch_size=100, peek_interval=100)
	tester.evaluate_model()

def test_Irvine_statistical():
	tester = ModelTester()
	tester.prepare_data('Irvine_data/statistical', class_list=DatasetLoader.upper_letters, fold=10, test_fold=0)
	tester.build_NN_model([500, 500, 500], activation=tf.nn.relu, dropout=0.5)											
	tester.train_model(algorithm=tf.train.AdamOptimizer, rate=1e-4, iteration=40000, batch_size=100, peek_interval=100)
	tester.evaluate_model()

def test_stanford_raw_softmax():
	tester = ModelTester()
	tester.prepare_data('Stanford_data/raw', class_list=DatasetLoader.lower_letters, fold=10, test_fold=0)
	tester.build_NN_model([], activation=tf.nn.relu, dropout=None)											
	tester.train_model(algorithm=tf.train.GradientDescentOptimizer, rate=5e-2, iteration=200000, batch_size=500, peek_interval=100)
	tester.evaluate_model()
	
def test_stanford_raw_1_layer():
	tester = ModelTester()
	tester.prepare_data('Stanford_data/raw', class_list=DatasetLoader.lower_letters, fold=10, test_fold=0)
	tester.build_NN_model([1000], activation=tf.nn.relu, dropout=None)											
	tester.train_model(algorithm=tf.train.GradientDescentOptimizer, rate=1e-2, iteration=50000, batch_size=100, peek_interval=100)
	tester.evaluate_model()
	
def test_stanford_raw_3_layer():
	tester = ModelTester()
	tester.prepare_data('Stanford_data/raw', class_list=DatasetLoader.lower_letters, fold=10, test_fold=0)
	tester.build_NN_model([500, 500, 500], activation=tf.nn.relu, dropout=0.5)											
	tester.train_model(algorithm=tf.train.GradientDescentOptimizer, rate=1e-4, iteration=50000, batch_size=100, peek_interval=100)
	tester.evaluate_model()

def test_stanford_raw_CNN():
	tester = ModelTester()
	tester.prepare_data('Stanford_data/raw', class_list=DatasetLoader.lower_letters, fold=10, test_fold=0)
	tester.build_CNN_model(28, 28, conv_layers=[32, 64], dense_layers=[1024], activation=tf.nn.relu, dropout=0.5)
	tester.train_model(algorithm=tf.train.AdamOptimizer, rate=2e-5, iteration=20000, batch_size=50, peek_interval=10, show_test=False)
	tester.evaluate_model()
	
def test_Stanford_statistical():
	tester = ModelTester()
	tester.prepare_data('Stanford_data/statistical', class_list=DatasetLoader.lower_letters, fold=10, test_fold=0)
	tester.build_NN_model([500, 500, 500], activation=tf.nn.relu, dropout=0.5)											
	tester.train_model(algorithm=tf.train.GradientDescentOptimizer, rate=1e-4, iteration=80000, batch_size=100, peek_interval=100)
	tester.evaluate_model()
	
def test_Stanford_diagonal():
	tester = ModelTester()
	tester.prepare_data('Stanford_data/diagonal', class_list=DatasetLoader.lower_letters, fold=10, test_fold=0)
	tester.build_NN_model([500, 500, 500], activation=tf.nn.relu, dropout=0.5)											
	tester.train_model(algorithm=tf.train.GradientDescentOptimizer, rate=1e-4, iteration=80000, batch_size=100, peek_interval=100)
	tester.evaluate_model()
	
def test_Stanford_centroid():
	tester = ModelTester()
	tester.prepare_data('Stanford_data/centroid', class_list=DatasetLoader.lower_letters, fold=10, test_fold=0)
	tester.build_NN_model([500, 500, 500], activation=tf.nn.relu, dropout=0.5)											
	tester.train_model(algorithm=tf.train.GradientDescentOptimizer, rate=1e-4, iteration=80000, batch_size=100, peek_interval=100)
	tester.evaluate_model()
	
def test_Stanford_hotspot():
	tester = ModelTester()
	tester.prepare_data('Stanford_data/hotspot', class_list=DatasetLoader.lower_letters, fold=10, test_fold=0)
	tester.build_NN_model([500, 500, 500], activation=tf.nn.relu, dropout=0.5)											
	tester.train_model(algorithm=tf.train.GradientDescentOptimizer, rate=1e-4, iteration=80000, batch_size=100, peek_interval=100)
	tester.evaluate_model()
	
test_Irvine_statistical()