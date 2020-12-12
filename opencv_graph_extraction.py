# trunc8 did this
import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Python scripts
import helper_functions, label_reader

@helper_functions.deep_copy_params
def trimWhitespace(img):
  height, width = img.shape
  margin = 5
  img_copy = 255*(img < 128).astype(np.uint8) # To invert the text to white
  coords = cv2.findNonZero(img_copy) # Find all non-zero points (text)
  x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
  return img[max(0, y-margin):min(height, y+h+margin), 
             max(0, x-margin):min(width, x+w+margin)] # Crop the image


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
def extractPlot(img, xaxis, yaxis):
  margin = 4
  img = img[yaxis['start']+margin:yaxis['end']-margin, 
            xaxis['start']+margin:xaxis['end']-margin]
  h,w = img.shape[:2]
  mask = np.zeros((h,w), np.uint8)

  # Transform to gray colorspace and threshold the image
  gray = img
  # gray = cv2.GaussianBlur(gray, (7,7), 0)
  _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

  # Perform opening on the thresholded image (erosion followed by dilation)
  kernel = np.ones((2,2),np.uint8)
  opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

  # Search for contours and select the biggest one and draw it on mask
  contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
  cnt = max(contours, key=cv2.contourArea)
  # x = list(map(lambda x: x[0][0], cnt))
  # y = list(map(lambda y: y[0][1], cnt))
  # plt.figure()
  # plt.plot(x,y)
  # plt.show()
  cv2.drawContours(mask, [cnt], 0, 255, -1)

  # Perform a bitwise operation
  res = cv2.bitwise_and(img, img, mask=mask)

  # Threshold the image again
  gray = img
  _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

  # Find all non white pixels
  non_zero = cv2.findNonZero(thresh)

  # Transform all other pixels in non_white to white
  for i in range(0, len(non_zero)):
      first_x = non_zero[i][0][0]
      first_y = non_zero[i][0][1]
      first = res[first_y, first_x]
      res[first_y, first_x] = 255

  # Display the image
  # cv2.imshow('original', img)
  # cv2.imshow('img', res)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def main():
  # Read args from terminal
  parser = argparse.ArgumentParser(description="Extract text from input image")
  parser.add_argument("-n", default="", help="image number")
  args = parser.parse_args()

  # Read file
  # filename = 'images/Line-Chart4.jpeg'
  filename = f'images/Line-Chart{args.n}.png'
  color_img = cv2.imread(filename)
  img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
  
  # Processing
  plt.figure("Pre-OCR")
  trimmed_img = trimWhitespace(img)
  xaxis, yaxis = getAxes(trimmed_img)
  zipped_x, zipped_y = label_reader.getLabels(trimmed_img, xaxis, yaxis)

  print(f"X-axis: {xaxis}\nY-axis: {yaxis}")
  print("X pixels:\n", list(zipped_x))
  print("Y pixels:\n", list(zipped_y))

  # extractPlot(trimmed_img, xaxis, yaxis)
  
  # to maximize
  # plt.get_current_fig_manager().full_screen_toggle()
  plt.show()


if __name__=='__main__':
  main()