from __future__ import division
import cv2
import numpy as np
import math


# Nearest neighbor interpolation 
def nn_interpolate(image, scale_factor):

	# Extract size
	(rows, cols, channels) = image.shape
	scaled_height = rows * scale_factor
	scaled_weight = cols * scale_factor

	# Compute ratio
	row_ratio = rows / scaled_height
	col_ratio = cols / scaled_weight

	row_position = np.floor(np.arange(scaled_height) * row_ratio).astype(int)
	column_position = np.floor(np.arange(scaled_weight) * col_ratio).astype(int)
	
	# Initialize scaled image
	scaled_image = np.zeros((scaled_height, scaled_weight, 3), np.uint8)

	for i in range(scaled_height):
		for j in range(scaled_weight):
			scaled_image[i, j] = image[row_position[i], column_position[j]]

	return scaled_image



# Bilinear interpolation
def bilinear_interpolate(image, scale_factor):

	(h, w, channels) = image.shape
	h2 = h  * scale_factor
	w2 = w  * scale_factor
	temp = np.zeros((h2, w2, 3), np.uint8)
	x_ratio = float((w - 1)) / w2;
	y_ratio = float((h - 1)) / h2;
	for i in range(1, h2 - 1): 
		for j in range(1 ,w2 - 1):
			x = int(x_ratio * j)
			y = int(y_ratio * i)
			x_diff = (x_ratio * j) - x
			y_diff = (y_ratio * i) - y
			a = image[x, y] & 0xFF
			b = image[x + 1, y] & 0xFF
			c = image[x, y + 1] & 0xFF
			d = image[x + 1, y + 1] & 0xFF
			blue = a[0] * (1 - x_diff) * (1 - y_diff) + b[0] * (x_diff) * (1-y_diff) + c[0] * y_diff * (1 - x_diff)   + d[0] * (x_diff * y_diff)
			green = a[1] * (1 - x_diff) * (1 - y_diff) + b[1] * (x_diff) * (1-y_diff) + c[1] * y_diff * (1 - x_diff)   + d[1] * (x_diff * y_diff)
			red = a[2] * (1 - x_diff) * (1 - y_diff) + b[2] * (x_diff) * (1-y_diff) + c[2] * y_diff * (1 - x_diff)   + d[2] * (x_diff * y_diff)
			temp[j, i] = (blue, green, red)

	return temp
			
def getBicPixelChannel(img,x,y,channel):
	if (x < img.shape[1]) and (y < img.shape[0]):
	    return img[y,x,channel] & 0xFF

	return 0


def Bicubic(img, rate):
	new_w = int(math.ceil(float(img.shape[1]) * rate))
	new_h = int(math.ceil(float(img.shape[0]) * rate))

	new_img = np.zeros((new_w, new_h, 3), np.uint8)

	x_rate = float(img.shape[1]) / new_img.shape[1]
	y_rate = float(img.shape[0]) / new_img.shape[0]

	C = np.zeros(5)

	for hi in range(new_img.shape[0]):
	    for wi in range(new_img.shape[1]):

	        x_int = int(wi * x_rate)
	        y_int = int(hi * y_rate)

	        dx = x_rate * wi - x_int
	        dy = y_rate * hi - y_int

	        for channel in range(new_img.shape[2]):
	            for jj in range(0,4):
	                o_y = y_int - 1 + jj
	                a0 = getBicPixelChannel(img, x_int, o_y, channel) 
	                d0 = getBicPixelChannel(img, x_int - 1, o_y, channel) - a0
	                d2 = getBicPixelChannel(img, x_int + 1, o_y, channel) - a0
	                d3 = getBicPixelChannel(img, x_int + 2, o_y, channel) - a0

	                a1 = -1./3 * d0 + d2 - 1./6 * d3
	                a2 = 1./2 * d0 + 1./2 * d2
	                a3 = -1./6 * d0 - 1./2 * d2 + 1./6 * d3
	                C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx

	            d0 = C[0] - C[1]
	            d2 = C[2] - C[1]
	            d3 = C[3] - C[1]
	            a0 = C[1]
	            a1 = -1. / 3 * d0 + d2 - 1. / 6 * d3
	            a2 = 1. / 2 * d0 + 1. / 2 * d2
	            a3 = -1. / 6 * d0 - 1. / 2 * d2 + 1. / 6 * d3
	            new_img[hi, wi, channel] = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy

	#return new_img
	return cv2.medianBlur(new_img, 3)



# Read image
image = cv2.imread('D:/project/image-processing/image/cameraman.jpg')
bil = bilinear_interpolate(image, 2)
nn = nn_interpolate(image, 2)
bic = Bicubic(image, 2)


cv2.imshow('o', image)
cv2.imshow('nn', nn)
cv2.imshow('bil', bil)
cv2.imshow('bic', bic)
cv2.waitKey(0)


