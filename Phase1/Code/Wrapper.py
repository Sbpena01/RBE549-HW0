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
import sklearn.cluster._kmeans

BASE_RESOLUTION = 7
BASE_GABOR_SCALE = 1.0

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

def createGaussianKernel(x, y, scale, elongation=1, order=1):
    kernel = 1/(2*np.pi*scale**2) * np.e**(-(x**2/scale+y**2/(scale*elongation))/(2*scale**2))
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

def solveGabor(x, y, wavelength, theta, phase_offset, sigma, aspect_ratio):
    x_prime = x*np.cos(theta) + y*np.sin(theta)
    y_prime = -x*np.sin(theta) + y*np.cos(theta)
    gabor = np.e**(-(x_prime**2+aspect_ratio**2*y_prime**2)/(2*sigma**2))*np.cos(2*np.pi*(x_prime/wavelength)+phase_offset)
    return gabor

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

def createRotations(input, num_rotations):
    rotation_bank = []
    rotation_increment = 360 / num_rotations
    theta=0
    for _ in range(num_rotations):
        rotated_filter = rotateAboutCenter(input, theta)
        rotation_bank.append(rotated_filter)
        theta += rotation_increment
    return rotation_bank

def scaleMatrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val == min_val:
        raise ValueError("Will Divide By Zero")
    scaled_matrix = ((matrix - min_val) / (max_val - min_val)) * 255  # Scale to [0, 255]
    scaled_matrix = np.clip(scaled_matrix, 0, 255)  # Ensure the values stay within [0, 255]
    return scaled_matrix

def generateDoGFilterBank(num_scale, num_rotations):
    filter_bank = []
    for scale in range(1, num_scale+1):
        size = BASE_RESOLUTION*scale
        size = size+1 if size % 2 == 0 else size 
        filter = np.zeros((size,size))
        center = int(size/2)+1
        for x in range(filter.shape[0]):
            for y in range(filter.shape[1]):
                # Transform so that (0,0) is in the center of the filter
                transformed_x = x+1 - center
                transformed_y = y+1 - center
                kernel = createGaussianKernel(transformed_x, transformed_y, scale, order=1)
                filter[x,y] = kernel
        # scaled_matrix = scaleMatrix(filter)
        image_row = createRotations(filter, num_rotations)
        for filter in image_row:
            filter_bank.append(filter)
    return filter_bank

def generateLMFilterBank(size="small"):
    if size == "small":
        scales = [1, np.sqrt(2), 2, 2*np.sqrt(2)]
    elif size == "large":
        scales = [np.sqrt(2), 2, 2*np.sqrt(2), 4]
    else:
        raise ValueError(f"Imcompatible size. Size must be 'small' or 'large'. Received: {size}")
    
    # This code block generates the first and second order derivative gaussian filters
    filter_bank = []
    for scale in scales[0:-1]:
        size = np.round(BASE_RESOLUTION * scale, decimals=0)  # I round to nearest integer since we cannot have 0.1 pixels
        size = int(size+1) if size % 2 == 0 else int(size)
        first_order_filter = np.zeros((size,size))
        second_order_filter = np.zeros((size,size))
        center = int(size/2)+1
        for x in range(first_order_filter.shape[0]):
            for y in range(first_order_filter.shape[1]):
                transformed_x = x+1 - center
                transformed_y = y+1 - center
                first_order_filter[x,y] = createGaussianKernel(transformed_x, transformed_y, scale, order=1, elongation=3)
                second_order_filter[x,y] = createGaussianKernel(transformed_x, transformed_y, scale, order=2, elongation=3)
        # scaled_first_order_filter = scaleMatrix(first_order_filter)
        first_order_rotations = createRotations(first_order_filter, 6)
        for filter in first_order_rotations:
            filter_bank.append(filter)
        # scaled_second_order_filter = scaleMatrix(second_order_filter)
        second_order_rotations = createRotations(second_order_filter, 6)
        for filter in second_order_rotations:
            filter_bank.append(filter)
    
    # Create the LOG filters
    log_scales = scales + [x*3 for x in scales]
    for scale in log_scales:
        size = np.round(BASE_RESOLUTION * scale, decimals=0)  # I round to nearest integer since we cannot have 0.1 pixels
        size = int(size+1) if size % 2 == 0 else int(size)
        log_filter = np.zeros((size,size))
        center = int(size/2)+1
        for x in range(log_filter.shape[0]):
            for y in range(log_filter.shape[1]):
                transformed_x = x+1 - center
                transformed_y = y+1 - center
                log_filter[x,y] = createGaussianKernel(transformed_x, transformed_y, scale, order=-1)
        # scaled_log_filter = scaleMatrix(log_filter)
        filter_bank.append(log_filter)

    # Create Gaussian filters
    for scale in scales:
        size = np.round(BASE_RESOLUTION * scale, decimals=0)  # I round to nearest integer since we cannot have 0.1 pixels
        size = int(size+1) if size % 2 == 0 else int(size)
        gaussian_filter = np.zeros((size,size))
        for x in range(gaussian_filter.shape[0]):
            for y in range(gaussian_filter.shape[1]):
                transformed_x = x+1 - center
                transformed_y = y+1 - center
                gaussian_filter[x,y] = createGaussianKernel(transformed_x, transformed_y, scale, order=0)
        # scaled_gaussian_filter = scaleMatrix(gaussian_filter)
        filter_bank.append(gaussian_filter)
    return filter_bank

def generateGaborFilterBank(num_rotations, num_scales):
    rotation_angles = np.linspace(0, np.pi, num_rotations)
    scales = np.arange(2.0, 2.0+(1.0*num_scales), 1.0)
    filter_bank = []
    for scale in scales:
        size = BASE_RESOLUTION*scale
        size = int(size+1) if size % 2 == 0 else int(size)
        filter = np.zeros((size,size))
        center = int(size/2)+1
        for theta in rotation_angles:
            for x in range(filter.shape[0]):
                for y in range(filter.shape[1]):
                    transformed_x = x+1 - center
                    transformed_y = y+1 - center
                    
                    #                   x, y, wavelength, theta, phase_offset, sigma, aspect_ratio
                    kernel = solveGabor(transformed_x, transformed_y, 4.0, theta, 0.0, scale, 1.0)
                    filter[x,y] = kernel
            # scaled_matrix = scaleMatrix(filter)
            filter_bank.append(filter)
    return filter_bank

def convolve2D(input: np.ndarray, filter: np.ndarray):
    # The filter should always be square for this assignment.
    if filter.shape[0] != filter.shape[1]:
        raise ValueError("Filter must be square.")
    result = np.zeros((input.shape[0], input.shape[1]))
    padding = 2*int(filter.shape[0]/2)
    padded_input = np.zeros((input.shape[0]+padding, input.shape[1]+padding))
    padded_input[1:-1, 1:-1] = input
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            region = padded_input[x:x+filter.shape[0], y:y+filter.shape[1]]
            resultant = np.sum(region * filter)
            result[x,y] = resultant
    return result   

def addPadding(input_matrix: np.ndarray, num_layers: int):
    output = input_matrix
    for _ in range(num_layers):
        padded = np.zeros((output.shape[0]+2, output.shape[1]+2))
        padded[1:-1, 1:-1] = output
        output = padded
    return output

def createImageFromFilterBank(filter_bank: list[np.ndarray], shape: tuple = None):
    # First, we need to find the largest array in the bank. This allows us to figure out
    # padding for the smaller arrays to be displayed.
    padded_bank = []
    largest_size = 0
    for filter in filter_bank:
        if filter.shape[0] > largest_size:
            largest_size = filter.shape[0]
    for filter in filter_bank:
        required_padding = largest_size - filter.shape[0]
        if required_padding == 0:
            scaled_filter = scaleMatrix(filter)
            padded_bank.append(scaled_filter)
            continue
        padded_filter = addPadding(filter, int(required_padding/2))
        padded_scaled_filter = scaleMatrix(padded_filter)
        padded_bank.append(padded_scaled_filter)
    result = np.hstack(padded_bank)
    return result
    

def main():
    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    dog_filter_bank = generateDoGFilterBank(2, 16)
    image = createImageFromFilterBank(dog_filter_bank, (1,1))
    cv2.imwrite("DoG_Filter_Bank.png", image)


    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """
    LMS = generateLMFilterBank()
    image = createImageFromFilterBank(LMS, (1,1))
    cv2.imwrite("LMS_Filter_Bank.png", image)

    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    gabor_filter_bank = generateGaborFilterBank(8, 5)
    image = createImageFromFilterBank(gabor_filter_bank, (1,1))
    cv2.imwrite("Gabor_Filter_Bank.png", image)


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
    
if __name__ == '__main__':
    main()
 


