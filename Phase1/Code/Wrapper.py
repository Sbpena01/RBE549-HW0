#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import scipy.ndimage as ndimage

DOG_STARTING_SCALE = 6

def getSobelOperators():
    gx = np.array([
		[-1, 0, 1],
  		[-2, 0, 2],
		[-1, 0, 1]
	])
    gy = np.array([
		[-1, -2, -1],
  		[0, 0, 0],
		[1, 2, 1]
	])
    return gx, gy

def createGaussianKernel(x, y, scale, order=1):
    kernel = 1/(2*np.pi*scale**2) * np.e**(-(x**2+y**2)/(2*scale**2))
    if order == 0:
        kernel_dx = kernel
    elif order == 1:
        kernel_dx = -(x/scale**2) * kernel
    elif order == 2:
        kernel_dx = (x**2/scale**4-1/scale**2) * kernel
    elif order == -1:  # Laplacian
        kernel_dx = ((x**2 + y**2)/scale**4 - 2/scale**2) * kernel
    else:
        raise ValueError('Gaussian order must be first or second order. Instead is: {order}')
    return kernel_dx

def rotateAboutCenter(matrix: np.ndarray, theta):
    #TODO: Improve this rotation function to remove the scipy include and properly rotate the matix.
    # size, _ = matrix.shape
    # center = int(size/2)+1
    # rotated_matrix = np.zeros((size,size))
    # for x in range(matrix.shape[0]):
    #     for y in range(matrix.shape[1]):
    #         centered_x = x+1 - center
    #         centered_y = y+1 - center
    #         new_x = int(center + np.cos(theta) * centered_x - np.sin(theta) * centered_y)-1
    #         new_y = int(center + np.sin(theta) * centered_x + np.cos(theta) * centered_y)-1
    #         if 0 <= new_x < size and 0 <= new_y < size:
    #             rotated_matrix[new_x, new_y] = matrix[x, y]
    rotated_matrix = ndimage.rotate(matrix, theta, reshape=False, mode='nearest')
    return rotated_matrix
            

def generateDoGFilterBank(num_scale, num_rotations):
    gx, gy = getSobelOperators()
    
    final_filters = []
    images = []
    for scale in range(1, num_scale+1):
        image_row = []
        size = DOG_STARTING_SCALE*scale
        size = size+1 if size % 2 == 0 else size  # Ensures size is always odd
        filter = np.zeros((size,size))
        center = int(size/2)+1
        
        for x in range(filter.shape[0]):
            for y in range(filter.shape[1]):
                # Transform so that (0,0) is in the center of the filter
                transformed_x = x+1 - center
                transformed_y = y+1 - center
                kernel = createGaussianKernel(transformed_x, transformed_y, scale, order=2)
                filter[x,y] = kernel
        
        rotation_increment = 360 / num_rotations
        theta=0
        for _ in range(num_rotations):
            rotated_filter = rotateAboutCenter(filter, theta)
            # Normalize the matrix values to the range [0, 255] with 0 mapped to 128
            # Scale to range [0, 255], with 0 mapped to 128
            min_val = np.min(rotated_filter)
            max_val = np.max(rotated_filter)
            scaled_matrix = ((rotated_filter - min_val) / (max_val - min_val)) * 255  # Scale to [0, 255]
            scaled_matrix = np.clip(scaled_matrix, 0, 255)  # Ensure the values stay within [0, 255]
            image_matrix = scaled_matrix.astype(np.uint8)
            image = cv2.cvtColor(image_matrix, cv2.COLOR_GRAY2BGR)
            image_row.append(image)
            theta += rotation_increment
        images.append(image_row)
    
    concat_image = cv2.hconcat(images[1])
    
    cv2.imwrite("dog_image.png", concat_image)

def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	generateDoGFilterBank(2, 16)


	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""



	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""


	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""


	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Brightness Map
	Perform brightness binning 
	"""


	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Color Map
	Perform color binning or clustering
	"""


	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
	pass
    
if __name__ == '__main__':
    main()
 


