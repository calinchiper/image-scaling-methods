from __future__ import division
import cv2
import numpy as np

# Nearest neighbor interpolation 
def nn_interpolate(image, scale_factor):

	# Extract size
	(rows, cols, channels) = image.shape
	scaled_height = rows * scale_factor
	scaled_weight = cols * scale_factor

	# Compute ratio
	row_ratio = rows / scaled_height
	col_ratio = cols / scaled_weight

	# Row interpolation 
	row_position = np.floor(np.arange(scaled_height) * row_ratio).astype(int)

	# Column interpolation
	column_position = np.floor(np.arange(scaled_weight) * col_ratio).astype(int)

	# Initialize scaled image
	scaled_image = np.zeros((scaled_height, scaled_weight, 3), np.uint8)

	for i in range(scaled_height):
		for j in range(scaled_weight):
			scaled_image[i, j] = image[row_position[i], column_position[j]]

	return scaled_image


# Read image
image = cv2.imread('D:/project/image-processing/image/cameraman.jpg')

dst = nn_interpolate(image, 5)

cv2.imshow('o', image)
cv2.imshow('s', dst)
cv2.waitKey(0)


