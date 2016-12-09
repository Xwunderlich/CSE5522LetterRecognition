import statistics as stat
import numpy as np
import math

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
  features = [0] * 16
  features[0] = minCol
  features[1] = minRow
  features[2] = maxCol - minCol
  features[3] = maxRow - minRow
  features[4] = count
  features[5] = sumX / count
  features[6] = sumY / count
  features[7] = sumX2 / count
  features[8] = sumY2 / count
  features[9] = sumXY / count
  features[10] = sumX2Y / count
  features[11] = sumY2X / count
  features[12] = countVE
  features[13] = sumYVE
  features[14] = countHE
  features[15] = sumXHE
  return features


# diagonal base feature extraction
def diagonal_features(image, **kwargs):
  
  zone_size = kwargs['zone_size']
    
  def zone_pixel_on(img, zone, row, col):
    zone_row = zone // (img.width // zone_size) * zone_size + row
    zone_col = zone % (img.width // zone_size) * zone_size + col
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

    
# hotspot feature extraction
def hotspot_features(image):
  num_directions = 8;
  max_d = 3;
  hotspots = []
  for x in range(3,image.width,5):
    for y in range(3,image.height,5):
      hotspots.append((x,y))
      
  def doDir(row,col,direction):
    if direction == 0:
      return row-1,col
    elif direction == 1:
      return row-1,col+1
    elif direction == 2:
      return row,col+1
    elif direction == 3:
      return row+1,col+1
    elif direction == 4:
      return row+1,col
    elif direction == 5:
      return row+1,col-1
    elif direction == 6:
      return row,col-1
    elif direction == 7:
      return row-1,col-1
    else:
      return row,col
  def distanceToOn(img,row,col,direction,max_d):
    for d in range(max_d):
      row,col = doDir(row,col,direction)
      if img.is_on(row,col):
        return 85*d
    return 255
    
  features = []
  for row,col in hotspots:
    features.append(row)
    features.append(col)
    for direction in range(num_directions):
      features.append(direction)
      features.append(distanceToOn(image,row,col,direction,max_d))
  return features
  

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
