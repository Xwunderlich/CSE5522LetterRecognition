from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class MNISTFeatureExtractor:
				
	def __init__(self, directory='MNIST_data/', one_hot=True):
		self.directory = directory
		self.mnist = None
		self.features = []
		self.labels = []
		self.normalize_factor = None
		self.count = 0
		self.one_hot = one_hot
		
	def __len__(self):
		return self.count
		
	def load(self):
		self.mnist = input_data.read_data_sets(self.directory, one_hot=self.one_hot)
	
	@staticmethod
	def pixel(img, row, col):
		return img[28 * row + col]
		
	@staticmethod
	def isOn(img, row, col):
		return pixel(img, row, col) >= 0.5
	
	def statistical_feature(self):
		
		def offLeft(img, row, col):
			return col == 0 or not isOn(img, row, col - 1)
			
		def offAbove(img, row, col):
			return row == 0 or not isOn(img, row - 1, col)
	
		def extract(image):
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
			
		for tr_image, tr_label in zip(self.mnist.train.images, self.mnist.train.labels):
			self.features.append(list(extract(tr_image)))
			self.labels.append(tr_label)
			self.count += 1
			print(self.count)
		for te_image, te_label in zip(self.mnist.test.images, self.mnist.test.labels):
			self.features.append(list(extract(te_image)))
			self.labels.append(te_label)
			self.count += 1
			print(self.count)
		self.features = np.array(self.features)
		self.labels = np.array(self.labels)
			
	def diagonal_extraction(self):
		def zone_pixel(img, zone, row, col):
			pass
			
	def normalize(self):
		abs_max_attr = np.amax(np.abs(self.features), axis=0)
		self.normalize_factor = 1024 / abs_max_attr
		for i in range(len(self.features)):
			self.features[i] *= self.normalize_factor
			
	def output(self, filepath):
		with open(filepath, 'w') as out:
			for feature, label in zip(self.features, self.labels):
				out.write(str(label))
				for attr in feature:
					out.write(', ')
					out.write('{:.2f}'.format(attr))
				out.write('\n')
		

extractor = MNISTFeatureExtractor(one_hot=False)
extractor.load()
extractor.statistical_feature()
extractor.normalize()
extractor.output('MNIST.data')
			
			