from modules.DatasetGenerator import *

def generate_stanford_diagonal(path='Stanford_data/diagonal.data', test=None):
	generator = DatasetGenerator()
	generator.load_Stanford()
	generator.extract(fd.diagonal_features, parameters={'zone_size': 2}, normalize_to=256, test=test)	
	generator.output(path, int_feature=True)
	
def generate_stanford_statistical(path='Stanford_data/statistical.data', test=None):
	generator = DatasetGenerator()
	generator.load_Stanford()
	generator.extract(fd.statistical_features, normalize_to=256, test=test)	
	generator.output(path, int_feature=False)
	
def generate_stanford_raw(path='Stanford_data/raw.data', test=None):
	generator = DatasetGenerator()
	generator.load_Stanford()
	generator.extract(None, test=test)
	generator.output(path, int_feature=True)

def generate_MNIST_diagonal(path='MNIST_data/diagonal.data', test=None):
	generator = DatasetGenerator()
	generator.load_MNIST()
	generator.extract(fd.diagonal_features, parameters={'zone_size': 4}, normalize_to=256, test=test)	
	generator.output(path, int_feature=True)	

def generate_MNIST_raw(path='MNIST_data/raw.data', test=None):
	generator = DatasetGenerator()
	generator.load_MNIST()
	generator.extract(None, normalize_to=256, test=test)	
	generator.output(path, int_feature=True)
	
generate_stanford_diagonal()