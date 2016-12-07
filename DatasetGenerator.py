from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import statistics as stat
import math

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
	
	def __init__(self):
		self.raw_features = []
		self.raw_labels = []
		self.features = []
		self.labels = []
		self.normalize_factor = None
		self.image_shape = None
		
	def __len__(self):
		return len(self.labels)
	
	# load the original MNIST data
	def load_MNIST(self, directory='MNIST_data/'):
		self.image_shape = (28, 28)
		mnist = input_data.read_data_sets(directory, one_hot=False)
		self.raw_features = np.concatenate((mnist.train.images, mnist.test.images))
		self.raw_labels = np.concatenate((mnist.train.labels, mnist.test.labels))
		
	def load_Stanford(self, filepath='Stanford_data/letter_recognition.data'):
		self.image_shape = (16, 8)
		with open(filepath, 'r') as source:
			while True:
				data = source.readline().rstrip()
				if data == str():
					break
				value = data.split('\t')
				label = value[1]
				feature = list(float(s) for s in value[6:])
				self.raw_labels.append(label)
				self.raw_features.append(feature)
	
	def extract(self, method=None, normalize=False, test=None):
		# for each image in training set, extract features and push it into self.features
		# also push the label into self.labels
		for count, (image, label) in enumerate(zip(self.raw_features, self.raw_labels)):
			if count == test:
				break
			extraction = image if method is None else list(method(self.Image(image, *self.image_shape)))
			self.features.append(extraction)
			self.labels.append(label)
			print(count)
			
		# convert to numpy array
		self.features = np.array(self.features)
		self.labels = np.array(self.labels)
		if test:
			print(self.features)
			print(self.labels)
		if normalize:
			self.normalize()
						
	# normalize the data to make features range from -1024 to 1024
	def normalize(self):
		abs_max_attr = np.amax(np.abs(self.features), axis=0)
		for i in range(len(abs_max_attr)):
			if abs_max_attr[i] != 0:
				abs_max_attr[i] = 256 / abs_max_attr[i]
		self.normalize_factor = abs_max_attr
		for i in range(len(self.features)):
			self.features[i] *= self.normalize_factor
			
	# write features and labels into a file
	def output(self, filepath, int_feature=False):
		with open(filepath, 'w') as out:
			for feature, label in zip(self.features, self.labels):
				out.write(str(label))
				for attr in feature:
					out.write(', ')
					out.write(str(int(attr)) if int else '{:.2f}'.format(attr))
				out.write('\n')
		

def statistical_features(image):
	count, countVE, countHE = 0, 0, 0
	minRow, minCol = None, None
	maxRow, maxCol = None, None
	sumX, sumY = 0, 0
	sumX2, sumY2, sumXY = 0, 0, 0
	sumX2Y, sumY2X = 0, 0
	sumXHE, sumYVE = 0, 0
	for pixel, (row, col) in image.on_pixels():
		x = col - image.width / 2
		y = row - image.height / 2
		count += 1
		sumX += x
		sumY += y
		sumX2 += x**2
		sumY2 += y**2
		sumXY += x*y
		sumX2Y += x**2*y
		sumY2X += y**2*x
		if image.is_off(row, col, offset=(0, -1)):
			countVE += 1
			sumYVE += y
		if image.is_off(row, col, offset=(-1, 0)):
			countHE += 1
			sumXHE += x
		if minRow is None or minRow > row:
			minRow = row
		if minCol is None or minCol > col:
			minCol = col
		if maxRow is None or maxRow < row:
			maxRow = row
		if maxCol is None or maxCol < col:
			maxCol = col
	_1 = minCol
	_2 = minRow
	_3 = maxCol - minCol
	_4 = maxRow - minRow
	_5 = count
	_6 = sumX / count
	_7 = sumY / count
	_8 = sumX2 / count
	_9 = sumY2 / count
	_10 = sumXY / count
	_11 = sumX2Y / count
	_12 = sumY2X / count
	_13 = countVE
	_14 = sumYVE
	_15 = countHE
	_16 = sumXHE
	return _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16


# diagonal base feature extraction
def diagonal_features(image):
	
	def find_box(img):
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
	
	def original_pos(row, col, minRow, minCol):
		return row + minRow, col + minCol
	
	def project_pos(row, col, size, height, width):
		p_row = int(row * height / size)
		p_col = int(col * width / size)
		return p_row, p_col
		
	def normalize_image(img):
		minRow, minCol, height, width = find_box(img)
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
		
	def normalize_image(img):
		minRow, minCol, height, width = find_box(img)
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
		
	def zone_pixel_on(img, zone, row, col):
		zone_row = zone // 7 * 4 + row
		zone_col = zone % 7 * 4 + col
		return pixel_value(img, zone_row, zone_col) >= 0.5
		
	def evaluate_zone(img, zone):
		diagonals = [0.0] * 7
		for row in range(4):
			for col in range(4):
				diagonals[row + col] += zone_pixel_on(img, zone, row, col)
		return stat.mean(diagonals)
		
	norm_img = image
	features = [0] * 49
	for zone in range(49):
		features[zone] = evaluate_zone(norm_img, zone)
	return features
	
		
extractor = DatasetGenerator()
extractor.load_Stanford()												# load mnist
extractor.extract(statistical_features, normalize=True, test=None)		# pass your feature extractor as the parameter
extractor.output('Stanford_statistical.data', int_feature=True)		# write to file
			
			