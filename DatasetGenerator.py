from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import statistics as stat
import math

class DatasetGenerator:
				
	def __init__(self):
		self.raw_features = []
		self.raw_labels = []
		self.features = []
		self.labels = []
		self.normalize_factor = None
		
	def __len__(self):
		return len(self.labels)
	
	# load the original MNIST data
	def load_MNIST(self, directory='MNIST_data/'):
		mnist = input_data.read_data_sets(directory, one_hot=False)
		self.raw_features = np.concatenate((mnist.train.images, mnist.test.images))
		self.raw_labels = np.concatenate((mnist.train.labels, mnist.test.labels))
		
	def load_Stanford(self, filepath='Stanford_data/letter_recognition.data'):
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
	
	def extract(self, method=None, normalize=False, test=False):
		# for each image in training set, extract features and push it into self.features
		# also push the label into self.labels
		for count, (image, label) in enumerate(zip(self.raw_features, self.raw_labels)):
			if test and count == 100:
				break
			extraction = image if method is None else list(method(image))
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
		

# helper method to get a pixel value by row and column
def pixel_value(img, row, col):
	return img[28 * row + col]

def statistical_features(image):
	def isOn(img, row, col):
		return pixel_value(img, row, col) >= 0.5

	def offLeft(img, row, col):
		return col == 0 or not isOn(img, row, col - 1)
		
	def offAbove(img, row, col):
		return row == 0 or not isOn(img, row - 1, col)
		
	count, countVE, countHE = 0, 0, 0
	minRow, minCol = None, None
	maxRow, maxCol = None, None
	sumX, sumY = 0, 0
	sumX2, sumY2, sumXY = 0, 0, 0
	sumX2Y, sumY2X = 0, 0
	sumXHE, sumYVE = 0, 0
	for i, pixel in enumerate(image):
		row, col = i // 28, i % 28
		x, y = col - 14, row - 14
		if isOn(image, row, col):
			count += 1
			sumX += x
			sumY += y
			sumX2 += x**2
			sumY2 += y**2
			sumXY += x*y
			sumX2Y += x**2*y
			sumY2X += y**2*x
			if offLeft(image, row, col):
				countVE += 1
				sumYVE += y
			if offAbove(image, row, col):
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
	def isOn(img, row, col):
		return pixel_value(img, row, col) >= 0.5
		
	def find_box(img):
		minRow, minCol = None, None
		maxRow, maxCol = None, None
		for i, pixel in enumerate(image):
			row, col = i // 28, i % 28
			if isOn(image, row, col):
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
extractor.load_Stanford()									# load mnist
extractor.extract(None, normalize=False, test=False)		# pass your feature extractor as the parameter
extractor.output('Stanford_raw.data', int_feature=True)	# write to file
			
			