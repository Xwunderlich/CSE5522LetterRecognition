from modules.DatasetGenerator import *

def generate_stanford_diagonal(path='Stanford_data/diagonal.data'):
	extractor = DatasetGenerator()
	extractor.load_Stanford()
	extractor.extract(fd.diagonal_features, normalize_to=256, test=None)	
	extractor.output(path, int_feature=True)
	
def generate_MNIST_raw(path='MNIST_data/raw.data'):
	extractor = DatasetGenerator()
	extractor.load_MNIST()
	extractor.extract(None, normalize_to=256, test=None)	
	extractor.output(path, int_feature=True)
	
generate_MNIST_raw()