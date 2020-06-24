#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
from matplotlib.pylab import plt
import math as m
from sklearn.cluster import KMeans
import os




def gaussian_kernel(size, mean_x, mean_y, var_x, var_y, order = 0):

	t = np.linspace(-(size - 1)/2, (size - 1)/2, size)

	x = np.exp(-(t - mean_x) ** 2 / (2 * var_x))
	y = np.exp(-(t - mean_y) ** 2 / (2 * var_y))

	if order == 1:
		x = -x * (t / var_x)
		# y = -y * (t / var_y)

	elif order == 2:
		x = x * ((t - mean_x) ** 2 - var_x)/(var_x ** 2)
		# y = y * ((t - mean_y) ** 2 - var_y) / (var_y ** 2)

	# x /= np.trapz(x)  # normalize the integral to 1
	# y /= np.trapz(y)  # normalize the integral to 1

	# make a 2-D kernel out of it
	kernel = x[:, np.newaxis] * y[np.newaxis, :]

	return kernel/np.max(kernel)


def Gaussian_bank(scales, orientations, size, plot=False, var_scale=0.25):
	variances = np.array([1, 2, 4, 8])

	var_vers = var_scale * np.linspace(size / 4, 3 * size / 4, scales)
	degrees_vec = np.linspace(0, 360 * (1 - 1 / orientations), orientations)
	gaussian_bank = np.array([[gaussian_kernel(size, 0, 0, variances[i], variances[i])] for i in range(4)])

	if plot:
		save_plot(DoG_filter_bank, 'DoG')

	return gaussian_bank



def LOG(size, var_x, var_y):
	filter = gaussian_kernel(size, 0, 0, var_x, var_y)
	return cv2.filter2D(filter, -1, Laplacian())


def sobel_filter():
	return np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])

def Laplacian():
	return np.array([[0, 1, 0],[1, -4, 1], [0, 1, 0]])


def DoG(size, var, rotation_degree):
	kernel = gaussian_kernel(size, 0, 0, var, var)
	sobel = sobel_filter()
	filter = cv2.filter2D(kernel, -1, sobel)

	return rotateImage(filter, rotation_degree, clock_wise=False)


def rotateImage(image, angle, clock_wise = True):
	rows, cols = image.shape[0], image.shape[1]
	matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 2 * (int(clock_wise) - 0.5))
	# fixing the rotation matrix to be n=in the (0 ,0)
	matrix[0, 2] += (matrix[0, 0] + matrix[0, 1] - 1) / 2
	matrix[1, 2] += (matrix[1, 0] + matrix[1, 1] - 1) / 2
	result = cv2.warpAffine(image, matrix, (cols, rows))
	return result

def save_plot(filter_bank, name):

	rows , cols = filter_bank.shape[0:2]

	plt.figure()
	sub = 1
	for row in range(rows):
		for col in range(cols):
			plt.subplot(rows, cols, sub)
			plt.imshow(filter_bank[row][col], cmap='gray')
			plt.axis('off')
			sub += 1

	plt.savefig(name)


def DoG_filter_bank(scales, orientations, size, plot=False, var_scale=0.25):

	var_vers = var_scale * np.linspace(size / 4, 3 * size / 4, scales)
	degrees_vec = np.linspace(0, 360 * (1 - 1/orientations), orientations)
	DoG_filter_bank = np.array([[DoG(size, var, degree) for degree in degrees_vec] for var in var_vers])

	if plot:
		save_plot(DoG_filter_bank, 'DoG')

	return DoG_filter_bank


def Leung_Malik_flter_bank(n_scales, n_orientations, size, type='small', plot=False):
	if type == 'small':
		variances = np.array([1, 2, 4, 8])

	elif type == 'large':
		variances = 2 * np.array([1, 2, 4, 8])

	LM_filter_bank = [[1 for _ in range(2 * n_orientations)] for _ in range(n_scales)]
	degrees_vec = np.linspace(0, 360 * (1 - 1 / n_orientations), n_orientations)

	for i in range(n_scales - 1):
		for j in range(n_orientations):

			LM_filter_bank[i][j] = rotateImage(gaussian_kernel(size, 0, 0, variances[i], 9 * variances[i], order=1),
											   degrees_vec[j])

			LM_filter_bank[i][j + 6] = rotateImage(gaussian_kernel(size, 0, 0, variances[i], 9 * variances[i], order=2),
											   degrees_vec[j])
	for i in range(n_scales):
		LM_filter_bank[n_scales - 1][i] = LOG(2 * size, variances[i], variances[i])
		LM_filter_bank[n_scales - 1][i + n_scales] = LOG(2 * size, 9 * variances[i], 9 * variances[i])
		LM_filter_bank[n_scales - 1][i + 2 * n_scales] = gaussian_kernel(size, 0, 0, variances[i], variances[i])


	LM_filter_bank = np.array(LM_filter_bank)

	if plot:
		save_plot(LM_filter_bank, 'Leung-Malik Filters')


	return LM_filter_bank


def Gabor_filter(sigma, Lambda, psi, gamma, size):

	sigma_x = sigma
	sigma_y = float(sigma) / gamma
	xmin, xmax = -(size - 1) / 2, (size - 1) / 2
	ymin, ymax = -(size - 1) / 2, (size - 1) / 2

	(y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

	return np.exp(-.5 * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x + psi)



def Gabor_filter_bank (scales, orientations, size, plot=False):

	gabor_filter_bank = [[1 for _ in range(orientations)] for _ in range(scales)]
	sigma_vec = np.array([6, 8, 14])
	lambda_vec = np.array([8, 10, 12])
	degrees_vec = np.linspace(0, 360 * (1 - 1 / orientations), orientations)

	for row in range(scales):
		for col in range(orientations):
			gabor_filter_bank[row][col] = rotateImage(Gabor_filter(sigma_vec[row], lambda_vec[row], 0, 0.75, size), degrees_vec[col])

	gabor_filter_bank = np.array(gabor_filter_bank)

	if plot:
		save_plot(gabor_filter_bank, 'Gabor_filter_bank')



	return gabor_filter_bank


def Texton_map(img):

	map = [[[[] for _ in range(3)] for _ in range(img.shape[1])] for _ in range(img.shape[0])]
	filters = DoG_filter_bank(2, 2, 35, var_scale=0.25)

	# filters = Gabor_filter_bank(3, 1, 17)
	# filters = Gaussian_bank(4, 1, 35)

	for row in range(filters.shape[0]):
		for col in range(filters.shape[1]):
			filtered_image = cv2.filter2D(img, -1, filters[row][col])

			for channel in range(3):
				for x in range(img.shape[0]):
					for y in range(img.shape[1]):
						map[x][y][channel].append(filtered_image[x][y][channel])


	return np.array(map)


def texture_ID(image, map, image_name):

	for ch in range(3):
		channel = map[:, :, ch, :]
		channel = channel.reshape(image.shape[0] * image.shape[1], map.shape[-1])
		km = KMeans(n_clusters=64).fit(channel)
		T_channel = km.labels_.reshape(image.shape[0], image.shape[1])
		image[:, :, ch] = T_channel

	plt.imshow(image/image.max())
	plt.axis('off')
	plt.savefig(image_name + '_texture_map')
	np.save(image_name + '_texture_map.npy', image)
	# plt.imshow(image/image.max())

def Brightness_map(image, image_name):
	# map = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3
	map = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	map = map.reshape(image.shape[0] * image.shape[1], 1)

	km = KMeans(n_clusters=16).fit(map)
	map = km.labels_.reshape(image.shape[0], image.shape[1])
	plt.imshow(map / map.max(), cmap='gray')
	plt.axis('off')
	plt.savefig(image_name + '_brightness_map')
	np.save(image_name + '_brightness_map.npy', map)



def Color_map(image, image_name):

	for ch in range(3):
		channel = image[:, :, ch]
		channel = channel.reshape(image.shape[0] * image.shape[1], 1)
		km = KMeans(n_clusters=16).fit(channel)
		T_channel = km.labels_.reshape(image.shape[0], image.shape[1])
		image[:, :, ch] = T_channel

	plt.imshow(image / image.max())
	plt.axis('off')
	plt.savefig(image_name + '_color_map')

	np.save(image_name + '_color_map.npy', image)



def Half_disk_masks(scales, orientations, plot = True):


	sizes = np.arange(9, 5 * (scales + 1),5)
	temp_degrees_vec = np.linspace(0, 360 * (1 - 1 / orientations), orientations)
	degrees_vec = []
	for index in range(int(orientations / 2)):
		degrees_vec.append(temp_degrees_vec[index])
		degrees_vec.append(temp_degrees_vec[index + int(orientations / 2)])

	half_disk_masks = [[1 for _ in range(orientations)] for _ in range(scales)]


	for i, size in enumerate(sizes):
		mask = np.zeros((size, size))
		for col in range(size):
			if col < size/2:
				for row in range(size):
					x = row - (size - 1) / 2
					y = col - (size - 1) / 2
					if (x ** 2 + y ** 2) < ((size ) / 2) * ((size ) / 2):
						mask[row][col] = 1
		for j, degree in enumerate(degrees_vec):
			half_disk_masks[i][j] = rotateImage(mask, degree)

	if plot:
		save_plot(np.array(half_disk_masks), 'Half_disk_masks')

	return half_disk_masks



def Gradient(image_name, map_name):
	mask_filters = Half_disk_masks(3, 8, plot=False)
	map = np.load(image_name + '_' + map_name + '_map.npy')
	# plt.imshow(map / map.max(), cmap='gray')
	# plt.savefig(image_name + '_map')
	gradients = []


	for row in range(3):
		for col in range(4):
			left_mask = mask_filters[row][2 * col]
			right_mask = mask_filters[row][2 * col + 1]
			chi_sqr_dist = map * 0
			k = map.max() + 1
			for bin in range(k):
				bin_chi_dist = map * 0
				temp = np.sign(-1 * (map - bin)) + 1

				g_i = cv2.filter2D(temp, -1, left_mask)
				h_i = cv2.filter2D(temp, -1, right_mask)

				num = np.square(h_i - g_i)
				denom = 1. / (g_i + h_i + 0.000005)
				bin_chi_dist = np.multiply(num, denom)


				# for x in range(temp.shape[0]):
				# 	for y in range(temp.shape[1]):
				# 		for z in range(temp.shape[2]):
				# 			if g_i[x][y][z] + h_i[x][y][z] != 0:
				# 				bin_chi_dist[x][y][z] = (g_i[x][y][z] - h_i[x][y][z]) ** 2 / (g_i[x][y][z] + h_i[x][y][z])


				chi_sqr_dist = chi_sqr_dist + bin_chi_dist / k

			gradients.append(chi_sqr_dist)

	gradient_map = np.mean(np.array(gradients), axis=0)

	if map_name == 'brightness':
		plt.imshow(gradient_map / gradient_map.max(), cmap='gray')
	else:
		plt.imshow(gradient_map / gradient_map.max())
	plt.axis('off')
	plt.savefig(image_name + '_' + map_name + '_gradient_map')
	np.save(image_name + '_' + map_name + '_gradient.npy', gradient_map)


def pb_lite(Sobel, Canny, image_name):
	B = np.load(image_name + '_brightness_gradient.npy')
	T = np.load(image_name + '_texture_gradient.npy')
	C = np.load(image_name + '_color_gradient.npy')

	B / B.max()
	T / T.max()
	C / C.max()



	w1 = 0.5
	w2 = 0.5
	features = (B + rgb2gray(T) + rgb2gray(C))/3
	average = w1 * Canny + w2 * Sobel
	pb_edge = features * rgb2gray(average)

	plt.imshow(pb_edge, cmap='gray')
	plt.savefig(image_name + '_pb_edge')

	s = 1


def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def edge_detection(root, image_path):

	image = cv2.imread(root + '/' + image_path)
	image_name = image_path.split('.')[0]


	"""
		Generate Texton Map
		Filter image using oriented gaussian filter bank
		"""

	map = Texton_map(image)

	"""
    Generate texture ID's using K-means clustering
    Display texton map and save image as TextonMap_ImageName.png,
    use command "cv2.imwrite('...)"
    """

	texture_ID(image, map, image_name)

	"""
    Generate Texton Gradient (Tg)
    Perform Chi-square calculation on Texton Map
    Display Tg and save image as Tg_ImageName.png,
    use command "cv2.imwrite(...)"
    """

	Gradient(image_name, 'texture')

	"""
    Generate Brightness Map
    Perform brightness binning 
    """

	Brightness_map(image, image_name)

	"""
    Generate Brightness Gradient (Bg)
    Perform Chi-square calculation on Brightness Map
    Display Bg and save image as Bg_ImageName.png,
    use command "cv2.imwrite(...)"
    """
	Gradient(image_name, 'brightness')

	"""
    Generate Color Map
    Perform color binning or clustering
    """
	Color_map(image, image_name)

	"""
    Generate Color Gradient (Cg)
    Perform Chi-square calculation on Color Map
    Display Cg and save image as Cg_ImageName.png,
    use command "cv2.imwrite(...)"
    """

	Gradient(image_name, 'color')

	"""
    Read Sobel Baseline
    use command "cv2.imread(...)"
    """

	os.chdir("..")
	cwd = os.getcwd()
	os.chdir(cwd + "/Code")

	Sobel = cv2.imread(cwd + "/BSDS500/SobelBaseline/" + image_name + ".png")

	"""
    Read Canny Baseline
    use command "cv2.imread(...)"
    """

	Canny = cv2.imread(cwd + "/BSDS500/CannyBaseline/" + image_name + ".png")

	"""
    Combine responses to get pb-lite output
    Display PbLite and save image as PbLite_ImageName.png
    use command "cv2.imwrite(...)"
    """

	pb_lite(Sobel, Canny, image_name)


def main():


	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""

	DoG_filter_bank(2, 5, 49, plot=True)





	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	Leung_Malik_flter_bank(4, 6, 35, plot=True)




	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	Gabor_filter_bank(3, 8, 69, plot=True)


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""

	Half_disk_masks(5, 8)


	os.chdir("..")
	cwd = os.getcwd() + "/BSDS500/Images"
	os.chdir(os.getcwd() + "/Code")
	for root, dirs, files in os.walk(cwd):
		for image in files:
			edge_detection(root, image)


if __name__ == '__main__':
	main()