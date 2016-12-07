import numpy as np
import random

class DatasetLoader:
	class DataPointer:
		def __init__(self, features, labels, indices):
			if len(features) != len(labels):
				raise ValueError('Size not match')
			self.all_features = features
			self.all_labels = labels
			self.indices = indices
			features, labels = list(), list()
			for i in indices:
				features.append(self.all_features[i])
				labels.append(self.all_labels[i])
			self.features = np.array(features)
			self.labels = np.array(labels)
			self.last = 0
		
		def __len__(self):
			return len(self.indices)
			
		def random_batch(self, batch_size=None):
			if batch_size is None:
				batch_size = len(self)
			features = list()
			labels = list()
			random.shuffle(self.indices)
			indices = self.indices[:batch_size]
			for i in indices:
				features.append(self.all_features[i])
				labels.append(self.all_labels[i])
			return np.array(features), np.array(labels)
			
		def next_batch(self, batch_size=None):
			if batch_size is None:
				batch_size = len(self)
			end = self.last + batch_size
			if end > len(self):
				end %= len(self) 
				features = np.concatenate((self.features[self.last:], self.features[:end]))
				labels = np.concatenate((self.labels[self.last:], self.labels[:end]))
			else:
				features = self.features[self.last:end]
				labels = self.labels[self.last:end]
			self.last = end
			return np.array(features), np.array(labels)
			
	digits = list(str(i) for i in range(10))
	letters = list(chr(u) for u in range(ord('A'), ord('Z')+1))
	
	def __init__(self, filepath, class_list):
		self.filepath = filepath
		self.class_list = class_list
		self.labels = None
		self.features = None
		self.train = None
		self.test = None
		
	def __len__(self):
		return self.features.shape[0]
		
	@property
	def attr_count(self):
		return self.features.shape[1]
		
	@property
	def class_count(self):
		return len(self.class_list)
		
	def load(self, one_hot=True):
		print('Loading Dataset...')
		labels = list()
		features = list()
		with open(self.filepath) as file:
			while True:
				data = file.readline().rstrip()
				if data == str():
					break
				value = data.split(',')
				label = self.class_list.index(value[0])
				if one_hot:
					one_hot_label = [0.0] * len(self.class_list)
					one_hot_label[label] = 1.0
					label = one_hot_label
				feature = list(float(s) for s in value[1:])
				labels.append(label)
				features.append(feature)
		self.labels = np.array(labels)
		self.features = np.array(features)
		print('Done Loading...')
		
	def fold(self, k, test_fold):
		test_start, test_end = int(len(self) * test_fold / k), int(len(self) * (test_fold + 1) / k)
		test_indices = list(range(test_start, test_end))
		train_indices = list(range(test_start)) + list(range(test_end, len(self)))
		self.test = self.DataPointer(self.features, self.labels, test_indices)
		self.train = self.DataPointer(self.features, self.labels, train_indices)
