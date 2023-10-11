from PIL import Image
import numpy as np
import pandas as pd
from numpy.linalg import eig

def cov_calculator(x,y): # this is used to calculate covariance of two vectors x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov = 0
    for i in range(len(x)):
        cov += (x[i] - x_mean)*(y[i] - y_mean)
    return cov/(len(x)-1)


def corr_calculator(x,y):
    return cov_calculator(x,y)/(np.sqrt(cov_calculator(x,x))*np.sqrt(cov_calculator(y,y)))


def cov_matrix_generator(big_mat): 
    cov_mat = np.zeros((len(big_mat),len(big_mat)))
    for i in range(len(big_mat)):
        for j in range(len(big_mat)):
            cov_mat[i][j] = cov_calculator(big_mat[i],big_mat[j])
        print(f'done with {i} rows')
    return cov_mat

im = Image.open('C:/Users/anupa/Downloads/jkl.jpg')

gray_image = im.convert('L')   # Convert the image to grayscale
gray_image.show() # Show the grayscale image

pix_val = list(gray_image.getdata()) # Convert the image to a 2D array
width, height = gray_image.size
pix_val = [pix_val[i * width:(i + 1) * width] for i in range(height)]

new_img = cov_matrix_generator(pix_val) # Generate the covariance matrix
print(np.shape(new_img))   
# Storing the currently obtained covariance matrix in a file, 
# so that we can use this value directly later on and 
# prevent the time needed to compute this covariance matrix
with open('C:/Users/anupa/Downloads/jkl.txt','w+') as f: 
    f.write("\n".join(map(str, new_img)))

# file_path = 'C:/Users/anupa/Downloads/abc.txt'
# with open(file_path, 'r') as file:
#     data = file.readlines()

# Convert the split strings to floating-point numbers
# new_img = np.array([line.strip('[]\n') for line in data])
# print(np.shape(new_img))    
# new_img = new_img.reshape((668, 668))
# print(np.shape(new_img))    
# print(new_img)

# Computed covariance matrix using numpy to match with our values and both of them matched
# cov_matrix = np.cov(pix_val) 
# print(np.shape(cov_matrix))   

# Code to convert the nparray to a RGB image
# PIL_image = Image.fromarray(np.uint8(new_img)).convert('RGB')
# PIL_image = Image.fromarray(new_img.astype('uint8'), 'RGB')

w,v=eig(new_img)
print('Eigen value:', w)
print('Eigen vector', v)
print(np.shape(w))  
print(np.shape(v))  
