import statistics as stat
import numpy as np

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
	
	zone_size=4
		
	def zone_pixel_on(img, zone, row, col):
		zone_row = zone // (img.width // 4) * 4 + row
		zone_col = zone % (img.width // 4) * 4 + col
		return img.is_on(zone_row, zone_col)
		
	def evaluate_zone(img, zone):
		diagonals = [0.0] * (zone_size * 2 - 1)
		for row in range(zone_size):
			for col in range(zone_size):
				diagonals[row + col] += zone_pixel_on(img, zone, row, col)
		return stat.mean(diagonals)
		
	norm_img = image
	num_zones = (norm_img.width // zone_size) * (norm_img.height // zone_size)
	features = [0] * num_zones
	for zone in range(num_zones):
		features[zone] = evaluate_zone(norm_img, zone)
	return features
	
		