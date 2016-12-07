from modules.DatasetGenerator import *

def generate_stanford_diagonal(path='Stanford_data/diagonal.data'):
	generator = DatasetGenerator()
	generator.load_Stanford()
	generator.extract(fd.diagonal_features, parameters={'zone_size': 2}, normalize_to=256, test=None)	
	generator.output(path, int_feature=True)
	
def generate_MNIST_raw(path='MNIST_data/raw.data'):
	generator = DatasetGenerator()
	generator.load_MNIST()
	generator.extract(None, normalize_to=256, test=None)	
	generator.output(path, int_feature=True)
	
generate_stanford_diagonal()