# trunc8 did this
import csv
import cv2
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import numpy as np

# Python scripts
import helper_functions


def getAxes(img):
  height, width = img.shape

  # Horizontal and vertical edges
  sobelx = np.uint8(np.absolute(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)))
  sobely = np.uint8(np.absolute(cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)))
  
  # Opening operations to bring out x and y axes
  kernelx = np.ones((height//10, 1), np.uint8)
  openedx = cv2.morphologyEx(sobelx, cv2.MORPH_OPEN, kernelx)
  kernely = np.ones((1, height//10), np.uint8)
  openedy = cv2.morphologyEx(sobely, cv2.MORPH_OPEN, kernely)

  # processing opened x to get Y-axis
  cropped_openedx = openedx[:, 0:width//2] # left half of image
  cropped_openedx = 255*(cropped_openedx > 110).astype(np.uint8)
  sum_along_columns = np.sum(cropped_openedx, axis = 0)
  yaxis_col = np.argmax(sum_along_columns)
  yaxis = {
    'start': helper_functions.findFirstNonZeroElement(cropped_openedx[:,yaxis_col]),
    'end': helper_functions.findLastNonZeroElement(cropped_openedx[:,yaxis_col])
  }

  # processing opened y to get X-axis
  cropped_openedy = openedy[height//2:height, :] # bottom half of image
  cropped_openedy = 255*(cropped_openedy > 110).astype(np.uint8)
  sum_along_rows = np.sum(cropped_openedy, axis = 1)
  xaxis_row = np.argmax(sum_along_rows)
  xaxis = {
    'start': helper_functions.findFirstNonZeroElement(cropped_openedy[xaxis_row, :]),
    'end': helper_functions.findLastNonZeroElement(cropped_openedy[xaxis_row, :])
  }

  # Plotting
  plt.subplot(3,3,1),plt.imshow(img,cmap = 'gray')
  plt.title('Cropped'), plt.xticks([]), plt.yticks([])
  plt.subplot(3,3,2),plt.imshow(sobelx,cmap = 'gray')
  plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
  plt.subplot(3,3,3),plt.imshow(sobely,cmap = 'gray')
  plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
  plt.subplot(3,3,4),plt.imshow(openedx,cmap = 'gray')
  plt.title('Opened Sobel X'), plt.xticks([]), plt.yticks([])
  plt.subplot(3,3,5),plt.imshow(openedy,cmap = 'gray')
  plt.title('Opened Sobel Y'), plt.xticks([]), plt.yticks([])
  plt.subplot(3,3,6),plt.imshow(cropped_openedx,cmap = 'gray')
  plt.title('Y axis'), plt.xticks([]), plt.yticks([])
  plt.subplot(3,3,7),plt.imshow(cropped_openedy,cmap = 'gray')
  plt.title('X axis'), plt.xticks([]), plt.yticks([])

  return [xaxis, yaxis]


@helper_functions.deep_copy_params
def extractPlot(img, xaxis, yaxis, m_x, b_x, m_y, b_y):
  step_size = 1
  margin = 10
  x_start = xaxis['start']+margin
  y_start = yaxis['start']+margin
  img = img[y_start:yaxis['end']-margin, 
            x_start:xaxis['end']-margin]
  h,w = img.shape

  # Remove noise
  kernel = np.ones((5,5),np.uint8)
  opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
  # Invert during thresholding
  _, thresh = cv2.threshold(opening,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  
  with open('graph_coordinates.csv', mode='w') as coordinates_file:
    coordinates_writer = csv.writer(coordinates_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for x_pixel in range(0, w, step_size):
      index1 = helper_functions.findFirstNonZeroElement(thresh[:,x_pixel])
      if index1 != -1:
        index2 = helper_functions.findLastNonZeroElement(thresh[:,x_pixel])
        y_pixel = (index1 + index2)/2
        x_coord = round(m_x*(x_start+x_pixel)+b_x, 2)
        y_coord = round(m_y*(y_start+y_pixel)+b_y, 2)
        coordinates_writer.writerow([x_coord, y_coord])

  # Display the image
  # cv2.imshow('original', img)
  # cv2.imshow('opened', opening)
  # cv2.imshow('img', thresh)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def mapPixelToCoordinate(coord):
  # Fitting line through the (pixels, cleaned_labels) data
  m, b = helper_functions.bestFitSlopeAndIntercept(pixels, cleaned_labels)
  return m, b