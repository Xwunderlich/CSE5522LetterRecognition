import statistics as stat
import numpy as np
import math

def centroid_features(image):

	def centroid_finder(img, zone_size, position_x, position_y):
		row_mass = 0
		col_mass = 0
		total_mass = 0
		for row in range(zone_size[0]):
			for col in range(zone_size[1]):
				row_mass += row * img[row, col]
				col_mass += col * img[row, col]
				total_mass += img[row, col]
		if total_mass == 0:
			centroid = 0
		else:
			centroid = position_x + int(round(row_mass / total_mass)), position_y + int(round(col_mass / total_mass))
		return centroid

	def compute_pixel_distance(pixel, centroid):
		distance = math.sqrt((pixel[0] - centroid[0]) ** 2 + (pixel[1] - centroid[1]) ** 2)
		return distance

	def average_pixel_distance_to_centroid(centroid, zone_size, zone_position):
		num_pixels = zone_size[0] * zone_size[1]
		distance = 0
		for i in range(zone_size[0]):
			for j in range(zone_size[1]):
				distance = distance + compute_pixel_distance((i + zone_position[0], j + zone_position[1]), centroid)
		average_distance = distance / num_pixels
		return average_distance




	img = image.image
	img = np.array(img)
	if img.shape == (784,):
		img = img.reshape(28,28)
		num_rows, num_cols = img.shape
		zone_size = [int(num_rows /4), int(num_cols /4)]
	else:	
		img = img.reshape(16,8)
		num_rows, num_cols = img.shape
		zone_size = [int(num_rows /4), int(num_cols /2)]

	n = int(num_rows * num_cols / (zone_size[0] * zone_size[1]))

	image_centroid = centroid_finder(img, img.shape, 0, 0)
	feature_counter = 0
	features = [0] * n * 2

	num1 = int(num_rows / zone_size[0])
	num2 = int(num_cols / zone_size[1])
	for i in range(num1):
		for j in range(num2):
			start_row = zone_size[0] * i
			finish_row = zone_size[0] * i + zone_size[0]
			start_col = zone_size[1] * j
			finish_col = zone_size[1] * j + zone_size[1]
			zone = img[start_row:finish_row, start_col:finish_col]
			zone_position = [start_row, start_col]

			# compute distance from every pixel in the zone to the image centroid
			average_pixel_distance = average_pixel_distance_to_centroid(image_centroid, zone_size, zone_position)
			features[feature_counter] = average_pixel_distance
			feature_counter += 1
			# Find Zone Centroid
			zone_centroid = centroid_finder(zone, zone_size, start_row, start_col)
			# If zone is empty(indicated by zone_centroid ==0), the corresponding feature is 0.
			if zone_centroid == 0:
				features[feature_counter] = 0
				feature_counter += 1
			else:
				# compute distance from every pixel in the zone to the zone centroid
				average_zone_pixel_distance_to_zone_centroid = average_pixel_distance_to_centroid(zone_centroid,
																								  zone_size,
																								  zone_position)
				features[feature_counter] = average_zone_pixel_distance_to_zone_centroid
				feature_counter += 1
	return features
