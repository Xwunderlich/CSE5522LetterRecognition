from modules.DatasetGenerator import *

def generate_stanford_diagonal(directory='Stanford_data/', name='diagonal', test=None):
  generator = DatasetGenerator()
  generator.load_Stanford()
  generator.extract(fd.diagonal_features, parameters={'zone_size': 2}, normalize_to=256, test=test) 
  generator.output(directory, name, int_feature=True)
  
def generate_stanford_statistical(directory='Stanford_data/', name='statistical', test=None):
  generator = DatasetGenerator()
  generator.load_Stanford()
  generator.extract(fd.statistical_features, normalize_to=256, test=test) 
  generator.output(directory, name, int_feature=False)

def generate_stanford_hotspot(directory='Stanford_data/', name='hotspot', test=None):
  generator = DatasetGenerator()
  generator.load_Stanford()
  generator.extract(fd.hotspot_features, normalize_to=None, test=test) 
  generator.output(directory, name, int_feature=False, chunks=10)
  
def generate_stanford_centroid(directory='Stanford_data/', name='centroid', test=None):
  extractor = DatasetGenerator()
  extractor.load_MNIST()
  extractor.extract(fd.centroid_features, normalize_to=256, test=test)  
  extractor.output(directory, name, int_feature=True)
  
def generate_stanford_raw(directory='Stanford_data/', name='raw', test=None):
  generator = DatasetGenerator()
  generator.load_Stanford()
  generator.extract(None, test=test)
  generator.output(directory, name, int_feature=True)

def generate_MNIST_diagonal(directory='MNIST_data/', name='diagonal', test=None):
  generator = DatasetGenerator()
  generator.load_MNIST()
  generator.extract(fd.diagonal_features, parameters={'zone_size': 4}, normalize_to=256, test=test) 
  generator.output(directory, name, int_feature=True, chunks=3)  
  
def generate_MNIST_statistical(directory='MNIST_data/', name='statistical', test=None):
  generator = DatasetGenerator()
  generator.load_MNIST()
  generator.extract(fd.statistical_features, normalize_to=256, test=test) 
  generator.output(directory, name, int_feature=False)
  
def generate_MNIST_hotspot(directory='MNIST_data/', name='hotspot', test=None):
  extractor = DatasetGenerator()
  extractor.load_MNIST()
  extractor.extract(fd.hotspot_features, normalize_to=None, test=test)  
  extractor.output(directory, name, int_feature=True, chunks=10)

def generate_MNIST_centroid(directory='MNIST_data/', name='centroid', test=None):
  extractor = DatasetGenerator()
  extractor.load_MNIST()
  extractor.extract(fd.centroid_features, normalize_to=256, test=test)  
  extractor.output(directory, name, int_feature=True)

def generate_MNIST_raw(directory='MNIST_data/', name='raw', test=None):
  generator = DatasetGenerator()
  generator.load_MNIST()
  generator.extract(None, normalize_to=256, test=test)  
  generator.output(directory, name, int_feature=True, chunks=10)
  
generate_MNIST_centroid()
generate_MNIST_hotspot()
generate_stanford_diagonal()
generate_stanford_statistical()
generate_stanford_hotspot()
generate_stanford_centroid()
generate_stanford_raw()
