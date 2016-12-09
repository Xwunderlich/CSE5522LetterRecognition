from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import os
import FeatureDescriptor as fd

class DatasetGenerator:
	
	class Image:
		def __init__(self, image, height, width, threshold = 0.5):
			self.image = image
			self.height = height
			self.width = width
			self.threshold = threshold
			
		def __iter__(self):
			for index, pixel in enumerate(self.image):
				row, col = index // self.width, index % self.width
				yield pixel, (row, col)
				
		def on_pixels(self, threshold=None):
			threshold = self.threshold if threshold is None else threshold
			for index, pixel in enumerate(self.image):
				if pixel < threshold:
					continue
				row, col = index // self.width, index % self.width
				yield pixel, (row, col)
			
		def pixel(self, row, col, offset=(0,0)):
			o_row, o_col = row + offset[0], col + offset[1]
			if o_row < 0 or o_col < 0 or o_row >= self.height or o_col >= self.width:
				return 0
			return self.image[self.width * o_row + o_col]
			
		def is_on(self, row, col, threshold=None, offset=(0,0)):
			threshold = self.threshold if threshold is None else threshold
			return self.pixel(row, col, offset) >= threshold
			
		def is_off(self, row, col, threshold=None, offset=(0,0)):
			threshold = self.threshold if threshold is None else threshold
			return not self.is_on(row, col, threshold, offset)
		
		@property
		def bounding_box(self):
			minRow, minCol = None, None
			maxRow, maxCol = None, None
			for pixel, (row, col) in image:
				if image.is_on(row, col):
					if minRow is None or minRow > row:
						minRow = row
					if minCol is None or minCol > col:
						minCol = col
					if maxRow is None or maxRow < row:
						maxRow = row
					if maxCol is None or maxCol < col:
						maxCol = col
			height, width = maxRow - minRow, maxCol - minCol
			return minRow, minCol, height, width
			
		def normalized_image(self):
			def original_pos(row, col, minRow, minCol):
				return row + minRow, col + minCol
			def project_pos(row, col, size, height, width):
				p_row = int(row * height / size)
				p_col = int(col * width / size)
				return p_row, p_col
			minRow, minCol, height, width = img.bounding_box
			norm_img = [0] * 1600
			for i in range(1600):
				row, col = i // 40, i % 40
				p_row, p_col = project_pos(row, col, 40, height, width)
				o_row, o_col = original_pos(p_row, p_col, minRow, minCol)
				norm_img[i] = pixel_value(img, o_row, o_col)
				if col == 0:
					print()
				print('*' if norm_img[i] >= 0.5 else ' ', end='')
			return norm_img
	
	def __init__(self, extension='data'):
		self.raw_features = []
		self.raw_labels = []
		self.features = []
		self.labels = []
		self.normalize_factor = None
		self.image_shape = None
		self.extension = extension
		
	def __len__(self):
		return len(self.labels)
	
	# load the original MNIST data
	def load_MNIST(self, directory='data_source/MNIST/'):
		self.image_shape = (28, 28)
		mnist = input_data.read_data_sets(directory, one_hot=False)
		self.raw_features = np.concatenate((mnist.train.images, mnist.test.images))
		self.raw_labels = np.concatenate((mnist.train.labels, mnist.test.labels))
		
	def load_Stanford(self, filepath='data_source/Stanford/letter_recognition.data'):
		self.image_shape = (16, 8)
		labels = [None] * 10
		features = [None] * 10
		for i in range(10):
			labels[i], features[i] = [], []
		with open(filepath, 'r') as source:
			while True:
				data = source.readline().rstrip()
				if data == str():
					break
				value = data.split('\t')
				data_id, label, next_id, word_id, position, fold = value[0:6]
				feature = list(float(s) for s in value[6:])
				labels[int(fold)].append(label)
				features[int(fold)].append(feature)
		self.raw_labels = np.concatenate(labels)
		self.raw_features = np.concatenate(features)
		print(len(self.raw_labels), len(self.raw_features))
	
	def extract(self, method=None, parameters={}, normalize_to=None, test=None):
		# for each image in training set, extract features and push it into self.features
		# also push the label into self.labels
		for count, (image, label) in enumerate(zip(self.raw_features, self.raw_labels)):
			if count == test:
				break
			extraction = image if method is None else list(method(self.Image(image, *self.image_shape), **parameters))
			self.features.append(extraction)
			self.labels.append(label)
			print('count:', count)
			
		# convert to numpy array
		self.features = np.array(self.features)
		self.labels = np.array(self.labels)
		if test:
			print(self.features)
			print(self.labels)
		if normalize_to is not None:
			self.normalize(max=normalize_to)
						
	# normalize the data to make features range from -1 to 1
	def normalize(self, max=1):
		abs_max_attr = np.amax(np.abs(self.features), axis=0)
		for i in range(len(abs_max_attr)):
			if abs_max_attr[i] != 0:
				abs_max_attr[i] = max / abs_max_attr[i]
		self.normalize_factor = abs_max_attr
		for i in range(len(self.features)):
			self.features[i] *= self.normalize_factor
			
	# write features and labels into a file
	def output(self, directory, name, int_feature=False, chunks=1):
		def make_path(directory, name):
			path = os.path.join(directory, name)
			if os.path.exists(path):
				print('Cleaning old data...')
				for file in os.listdir(path):
					os.remove(os.path.join(path, file))
			else:
				os.makedirs(path)
			return path
		
		def get_filename(path, chunk):
			chunk_file = 'chunk' + '{:02}'.format(chunk) + '.' + self.extension
			return os.path.join(path, chunk_file)
		
		chunk_size = len(self) // chunks
		pointer = 0
		path = make_path(directory, name)
		for c in range(chunks):
			filepath = get_filename(path, c)
			print('Exporting chunk ({}/{})...'.format(c, chunks - 1))
			with open(filepath, 'w') as out:
				for index in range(chunk_size):
					if pointer + index == len(self):
						break
					feature, label = self.features[pointer + index], self.labels[pointer + index]
					out.write(str(label))
					for attr in feature:
						out.write(', ')
						out.write(str(int(attr)) if int_feature else '{:.2f}'.format(attr))
					out.write('\n')
				pointer += chunk_size
			
			