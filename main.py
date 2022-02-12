from operator import imatmul
from PIL.Image import new
import cv2
from time import time
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient
import csv


# def convolve2D(image, kernel, padding=2, strides=1):
#     # Cross Correlation, turn the kernel by 180 degrees
#     kernel = np.flipud(np.fliplr(kernel))

#     # Gather Shapes of Kernel + Image + Padding
#     xKernShape = kernel.shape[0]
#     yKernShape = kernel.shape[1]
#     xImgShape = image.shape[0]
#     yImgShape = image.shape[1]

#     # Shape of Output Convolution
#     xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
#     yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
#     output = np.zeros((xOutput, yOutput))

#     # Apply Equal Padding to All Sides
#     if padding != 0:
#         imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
#         imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
#         # print(imagePadded)
#     else:
#         imagePadded = image

#     # Iterate through image
#     for y in range(yImgShape):
#         # Exit Convolution
#         if y > yImgShape - yKernShape:
#             break
#         # Only Convolve if y has gone down by the specified Strides
#         if y % strides == 0:
#             for x in range(xImgShape):
#                 # Go to next row once kernel is out of bounds
#                 if x > xImgShape - xKernShape:
#                     break
#                 try:
#                     # Only Convolve if x has moved by the specified Strides
#                     if x % strides == 0:
#                         output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
#                 except:
#                     break
#     return output

# def hysterisis_recusive(image, weak_index, weakedge_row, weakedge_col, strong_index, strongedge_row, strongedge_col):
#     result = np.copy(image)
#     for i in range(strong_index):
#         result = find_connected_weak_edge(result, strongedge_row[i], strongedge_col[i])
    
#     for i in range(weak_index):
#         if result[weakedge_row[i], weakedge_col[i]] != 255:
#             result[weakedge_row[i], weakedge_col[i]] = 0
    
#     return result

# def find_connected_weak_edge(image, row, col):
#     M, N = image.shape
#     for i in range(-3, 3, 1):
#         for j in range(-3, 3, 1):
#             if (row+i > 0) and (col+j >0) and (row+i < M) and (col+j < N):
#                 image[int(row+i), int(col+j)] = 255
#                 image = find_connected_weak_edge(image, row+i, col+j)
#     return image

def convolution(image, kernel, average=False):
    print("Image Shape : {}".format(image.shape))
    print("Kernel Shape : {}".format(kernel.shape))

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    #padding
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image


    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))
    return output

def generate_gaussian(size, sigma=1):
    size = int(size) // 2  # floor divition operator '//' true divitrion operator '/'
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    result = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return result

def generate_sobel(direction):
    if direction.lower() == 'x':
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    elif direction.lower() == 'y':
        return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

def RGB2GRAY(image):
    # RGB weight: L = R * 299/1000 + G * 587/1000 + B * 114/1000
    rgb_weights = [0.2989, 0.587, 0.114]
    result = np.dot(image[...,:3], rgb_weights) # normalize the grayimage to 0.0 to 1.0 to display
    result = result.astype(np.uint8)
    return result

def gaussian_blur(image):
    kernel = generate_gaussian(size=5)
    result = convolution(image, kernel)
    return result

def sobel_dege_detector(image):
    x_kernel = generate_sobel('x')
    y_kernel = generate_sobel('y')
    Ix = convolution(image, x_kernel)
    Iy = convolution(image, y_kernel)
    gradient_magnitude = np.sqrt(np.square(Ix) + np.square(Iy))
    gradient_direction = np.arctan2(Iy, Ix)
    return gradient_direction, gradient_magnitude, Ix, Iy

def nms_interpolation(magnitude, dx, dy, sub_pixel=False):
    # use sub-pixel interpolation to determine the edge pixel
    mag = np.copy(magnitude)
    M, N = np.shape(mag)
    result = np.zeros((M,N))
    sub_pixel_location = []

    for i in range(1, M-1):
        for j in range(1, N-1):
            if mag[i, j] == 0:
                result[i, j] =0
            else:
                gradX = dx[i,j]
                gradY = dy[i,j]
                gradX_abs = np.abs(gradX)
                gradY_abs = np.abs(gradY)
                grad = mag[i,j]

                # Y gradient greater than X gradient
                if gradY_abs > gradY_abs:
                    if gradY == 0:
                        weight = 0
                    else:
                        weight = gradX_abs / gradY_abs
                    # |g1|g2|  |    |  |g2|g1|
                    # |  | g|  | or |  | g|  |
                    # |  |g4|g3|    |g3|g4|  |
                    g2 = mag[i-1, j]
                    g4 = mag[i+1, j]
                    if gradX * gradY > 0:
                        g1 = mag[i-1, j-1]
                        g3 = mag[i+1, j+1]
                    else:
                        g1 = mag[i-1, j+1]
                        g3 = mag[i+1, j-1]
                # X gradient greater than Y gradient
                else:
                    if gradX == 0:
                        weight = 0
                    else:
                        weight = gradY_abs / gradX_abs
                    # |  |  |g3|    |g1|  |  |
                    # |g2| g|g4| or |g2| g|g4|
                    # |g1|  |  |    |  |  |g3|
                    g2 = mag[i, j-1]
                    g4 = mag[i, j+1]
                    if gradX * gradY > 0:
                        g1 = mag[i+1, j-1]
                        g3 = mag[i-1, j+1]
                    else:
                        g1 = mag[i-1, j-1]
                        g3 = mag[i+1, j+1]
                
                gradTemp1 = weight * g1 + (1-weight) * g2
                gradTemp2 = weight * g3 + (1-weight) * g4
                if grad >= gradTemp1 and grad >= gradTemp2:
                    result[i,j] = grad
                else:
                    result[i,j] = 0

    return result

def double_threshold(image, th_low_ratio = 0.1, th_high_ratio = 0.3):
    # upper lower ratio is recommended to be between 2:1 or 3:1
    highThreshold = np.max(image) * th_high_ratio
    lowThreshold = np.max(image) * th_low_ratio
    
    M, N = image.shape
    result = np.zeros((M,N), dtype=np.int32)

    strongedge_row = np.zeros(M*N)
    strongedge_col = np.zeros(M*N)
    weakedge_row = np.zeros(M*N)
    weakedge_col = np.zeros(M*N)
    strong_index = 0
    weak_index = 0
    
    weak_value = np.int32(50)
    strong_value = np.int32(255)
    
    # strong_i, strong_j = np.where(image >= highThreshold)
    # weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))
    # zeros_i, zeros_j = np.where(image < lowThreshold)
    # result[strong_i, strong_j] = strong_value
    # result[weak_i, weak_j] = weak_value

    for i in range(M):
        for j in range(N):
            if image[i, j] > highThreshold:
                result[i, j] = strong_value
                strongedge_row[strong_index] = i
                strongedge_col[strong_index] = j
                strong_index += 1
            elif image[i, j] < lowThreshold:
                result[i, j] = 0
            else:
                result[i, j] = weak_value
                weakedge_row[weak_index] = i
                weakedge_col[weak_index] = j
                weak_index += 1

    return result, weak_index, weakedge_row, weakedge_col, strong_index, strongedge_row, strongedge_col

def hysterisis(image, weak_value=50, strong_value=255):
    M, N = image.shape
    # result = np.zeros((M,N))
    top2btm = np.copy(image)
    btm2top = np.copy(image)
    right2left = np.copy(image)
    left2right = np.copy(image)

    # probably needs to go from other directions?
    for i in range(1, M):
        for j in range(1, N):
            # check the 8 surroundings of the weak edge
            if(top2btm[i,j]==weak_value):
                if((top2btm[i-1, j-1:j+1] == strong_value).any()
                     or (top2btm[i, [j-1,j+1]] == strong_value).any()
                     or (top2btm[i+1, j-1:j+1] == strong_value).any()):
                    top2btm[i,j] = strong_value
                else:
                    top2btm[i,j] = 0

    for i in range(M-1, 1, -1):
        for j in range(N-1, 1, -1):
            # check the 8 surroundings of the weak edge
            if(btm2top[i,j]==weak_value):
                if((btm2top[i-1, j-1:j+1] == strong_value).any()
                     or (btm2top[i, [j-1,j+1]] == strong_value).any()
                     or (btm2top[i+1, j-1:j+1] == strong_value).any()):
                    btm2top[i,j] = strong_value
                else:
                    btm2top[i,j] = 0

    for i in range(1, M):
        for j in range(N-1, 0, -1):
            if(right2left[i,j]==weak_value):
                if((right2left[i-1, j-1:j+1] == strong_value).any()
                    or (right2left[i, [j-1,j+1]] == strong_value).any()
                    or (right2left[i+1, j-1:j+1] == strong_value).any()):
                    right2left[i,j] = strong_value
                else:
                    right2left[i,j] = 0

    for i in range(M-1, 0, -1):
        for j in range(1, N):
            if(left2right[i,j]==weak_value):
                if((left2right[i-1, j-1:j+1] == strong_value).any()
                    or (left2right[i, [j-1,j+1]] == strong_value).any()
                    or (left2right[i+1, j-1:j+1] == strong_value).any()):
                    left2right[i,j] = strong_value
                else:
                    left2right[i,j] = 0

    result = top2btm + btm2top + right2left + left2right
    result[result > 255] = 255
    return result

def fit_parabola(x1, y1, x2, y2, x3, y3):
    # y = ax^2+bx+c
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    a = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    b = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    c = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom
    x0 = (-1*b)/(2*a)
    return x0, y2

def canny(image, verbose=True):
    # 5 steps
    # 1.Noise reduction (guassian blur)
    img_gaussian = gaussian_blur(image)
    if verbose:
        cv2.imshow('gaussian_blur',img_gaussian.astype(np.uint8))
    
    # 2.edge enhancement (gradient caluclation)
    gradient_direction, gradient_magnitude, Ix, Iy = sobel_dege_detector(img_gaussian)
    if verbose:
        cv2.imshow('edge enhancement',gradient_magnitude.astype(np.uint8))

    # 3.Non-maximum suppression (pixel accuracy)
    img_nms = nms_interpolation(gradient_magnitude,dx=Ix, dy=Iy)
    if verbose:
        cv2.imshow('NMS_result', img_nms.astype(np.uint8))

    # 4.Double thresholding
    img_dt, weak_index, weakedge_row, weakedge_col, strong_index, strongedge_row, strongedge_col = double_threshold(img_nms)
    if verbose:
        cv2.imshow('double thresholding', img_dt.astype(np.uint8))

    # 5.Edge Tracking by Hysteresis
    final_result = hysterisis(img_dt)
    # final_result = hysterisis_recusive(img_dt, weak_index, weakedge_row, weakedge_col, strong_index, strongedge_row, strongedge_col)
    cv2.imshow('hysterisis', final_result.astype(np.uint8))

    return final_result

def compute_edge_points(partial_gradients, min_magnitude=0):
    gx, gy = partial_gradients
    rows, cols = gx.shape
    edges = []

    def mag(y, x):
        # sqrt(x^2+y^2) magnitude
        return np.hypot(gx[y, x], gy[y, x])

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):

            center_mag = mag(y, x)
            if center_mag < min_magnitude:
                continue

            left_mag = mag(y, x - 1)
            right_mag = mag(y, x + 1)
            top_mag = mag(y - 1, x)
            bottom_mag = mag(y + 1, x)

            theta_x, theta_y = 0, 0
            if (left_mag < center_mag >= right_mag) and abs(gx[y, x]) >= abs(gy[y, x]):
                theta_x = 1
            elif (top_mag < center_mag >= bottom_mag) and abs(gx[y, x]) <= abs(gy[y, x]):
                theta_y = 1

            if theta_x != 0 or theta_y != 0:
                a = mag(y - theta_y, x - theta_x)
                b = mag(y, x)
                c = mag(y + theta_y, x + theta_x)
                lamda = (a - c) / (2 * (a - 2 * b + c))
                ex = x + lamda * theta_x
                ey = y + lamda * theta_y
                edges.append([ex, ey])
                print('(%f, %f)' % (ex, ey))
    return edges

def canny_subpixel(image):
    # 1.resize
    M, N = image.shape
    img_resize = cv2.resize(image, (int(N/4), int(M/4)), interpolation=cv2.INTER_AREA)
    cv2.imshow('resized_image',img_resize.astype(np.uint8))

    # 2.Noise reduction (guassian blur)
    img_gaussian = gaussian_blur(img_resize)
    cv2.imshow('gaussian_blur_resized',img_gaussian.astype(np.uint8))
    
    # 3.edge enhancement (gradient caluclation)
    gradient_direction, gradient_magnitude, Ix, Iy = sobel_dege_detector(img_gaussian)
    cv2.imshow('edge enhancement_resized',gradient_magnitude.astype(np.uint8))

    # 4.sub pixel canny
    edgels = compute_edge_points((Ix, Iy),50)
    
    # 5.get the edge location

    # 6.cast again, resize it back to original image
    subpixel_result = np.zeros((M,N))
    for i in edgels:
        y = int(i[0]*4)
        x = int(i[1]*4)
        subpixel_result[x,y] = 255
    cv2.imshow('sub_pixel', subpixel_result)

    # 7. calculate accuracy
    canny_result = canny(image, verbose=True)
    accuracy = calculate_accuracy(canny_result, subpixel_result)
    print('the accuracy is: %f' % accuracy)
    return subpixel_result

def calculate_accuracy(canny, subpixel):
    M, N = subpixel.shape
    correct = 0
    incorrect = 0
    for i in range(M):
        for j in range(N):
            if subpixel[i,j] == 255:
                if (canny[i-3:i+3, j-3:j+3] == 255).any():
                    correct += 1
                else:
                    incorrect += 1
    
    accuracy = correct/(correct+incorrect)
    return accuracy

def read_image(image_name, image_ext, sub_pixel=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, image_name + '.' + image_ext)
    image = cv2.imread(image_path)
    cv2.imshow('original', image)

    if len(image.shape) == 3:
        gray_image = RGB2GRAY(image)
    elif len(image.shape) == 2:
        gray_image = image
    cv2.imshow('gray', gray_image)

    # # normalize the grayimage to 0.0 to 1.0 for display
    # gray_image /= 255. 
    # new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(new_gray)
    # # opencv image is brighter, probabaly due to the low precision for fast computation speed
    # print(gray_image) 

    # CANNY
    if sub_pixel:
        result = canny_subpixel(gray_image)
    else:
        result = canny(gray_image)
        # canny_cv =  cv2.Canny(np.uint8(image),200, 300)
        # cv2.imshow('canny', canny_cv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def realtime():
    camera = cv2.VideoCapture(0)
    kernel_gauss = generate_gaussian(5)
    while True:
        # Read the frame
        ret, frame= camera.read()
        # mirroing the image
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = canny(gray,kernel_gauss)
        # canny_cv =  cv2.Canny(np.uint8(frame),200, 300)
        # Display
        cv2.imshow('canny', img)
        cv2.imshow('frame', frame)
        #cv2.imshow('canny_opencv', canny_cv)
        # Stop if escape key is pressed
        k = cv2.waitKey(20) & 0xff
        if k==27:
            break

if __name__ == '__main__':
    # sys.setrecursionlimit(10000)
    # read_image('chessboard_hp', 'jpg', sub_pixel=False)
    read_image('chessboard_hp', 'jpg', sub_pixel=True)
    # read_image('lena', 'png', sub_pixel=True)
    # print(generate_gaussian(5))
    # gaussian separable. use 1D filter to reduce calculation time
    # print(cv2.getGaussianKernel(ksize=5,sigma=1) * cv2.getGaussianKernel(ksize=5,sigma=1).T)''