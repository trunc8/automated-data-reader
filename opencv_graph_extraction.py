# trunc8 did this
import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pytesseract

def findLastNonZeroElement(lst):
  return [index for index, item in enumerate(lst) if item != 0][-1]

def findFirstNonZeroElement(lst):
  return [index for index, item in enumerate(lst) if item != 0][0]

def getAxes(img):
  height = img.shape[0]
  width = img.shape[1]

  plt.subplot(3,3,1),plt.imshow(img,cmap = 'gray')
  plt.title('Cropped'), plt.xticks([]), plt.yticks([])

  # Horizontal and vertical edges
  sobelx = np.uint8(np.absolute(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)))
  sobely = np.uint8(np.absolute(cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)))
  
  plt.subplot(3,3,2),plt.imshow(sobelx,cmap = 'gray')
  plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
  plt.subplot(3,3,3),plt.imshow(sobely,cmap = 'gray')
  plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

  kernelx = np.ones((height//10, 1), np.uint8)
  openedx = cv2.morphologyEx(sobelx, cv2.MORPH_OPEN, kernelx)
  kernely = np.ones((1, height//10), np.uint8)
  openedy = cv2.morphologyEx(sobely, cv2.MORPH_OPEN, kernely)

  plt.subplot(3,3,4),plt.imshow(openedx,cmap = 'gray')
  plt.title('Opened Sobel X'), plt.xticks([]), plt.yticks([])
  plt.subplot(3,3,5),plt.imshow(openedy,cmap = 'gray')
  plt.title('Opened Sobel Y'), plt.xticks([]), plt.yticks([])

  # processing opened x to get Y-axis
  openedx = cv2.cvtColor(openedx,cv2.COLOR_BGR2GRAY)
  cropped_openedx = openedx[:, 0:width//2] # left half of image
  mask = cropped_openedx > 110
  cropped_openedx[mask] = 255
  cropped_openedx[~mask] = 0
  plt.subplot(3,3,6),plt.imshow(cropped_openedx,cmap = 'gray')
  plt.title('Y axis'), plt.xticks([]), plt.yticks([])

  sum_along_columns = np.sum(cropped_openedx, axis = 0)
  yaxis_col = np.argmax(sum_along_columns)
  yaxis = {
    'start': findFirstNonZeroElement(cropped_openedx[:,yaxis_col]),
    'end': findLastNonZeroElement(cropped_openedx[:,yaxis_col])
  }

  # processing opened y to get X-axis
  openedy = cv2.cvtColor(openedy,cv2.COLOR_BGR2GRAY)
  cropped_openedy = openedy[height//2:height, :] # bottom half of image
  mask = cropped_openedy > 110
  cropped_openedy[mask] = 255
  cropped_openedy[~mask] = 0
  plt.subplot(3,3,7),plt.imshow(cropped_openedy,cmap = 'gray')
  plt.title('X axis'), plt.xticks([]), plt.yticks([])

  sum_along_rows = np.sum(cropped_openedy, axis = 1)
  xaxis_row = np.argmax(sum_along_rows)
  xaxis = {
    'start': findFirstNonZeroElement(cropped_openedy[xaxis_row, :]),
    'end': findLastNonZeroElement(cropped_openedy[xaxis_row, :])
  }

  return [xaxis, yaxis]

def getLabels(img, xaxis, yaxis):
  print('Y axis text')
  behind_yaxis = img[:,:xaxis['start']]
  text = pytesseract.image_to_string(behind_yaxis, lang="eng", config="--psm 11")
  print(text)
  plt.subplot(3,3,8),plt.imshow(behind_yaxis,cmap = 'gray')
  plt.title('Behind Y axis'), plt.xticks([]), plt.yticks([])

  print('X axis text')
  below_xaxis = img[yaxis['end']:]
  text = pytesseract.image_to_string(below_xaxis, lang="eng", config="--psm 11")
  print(text)
  plt.subplot(3,3,9),plt.imshow(below_xaxis,cmap = 'gray')
  plt.title('Below X axis'), plt.xticks([]), plt.yticks([])

def main():
  # Read args from terminal
  parser = argparse.ArgumentParser(description="Extract text from input image")
  parser.add_argument("-n", default="", help="image number")
  args = parser.parse_args()

  # Read file
  filename = f'images/Line-Chart{args.n}.png'
  img = cv2.imread(filename)
  height = img.shape[0]
  width = img.shape[1]
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # Trim whitespace
  margin = 10
  gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
  coords = cv2.findNonZero(gray) # Find all non-zero points (text)
  x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
  img = img[max(0, y-margin):min(height, y+h+margin), 
            max(0, x-margin):min(width, x+w+margin)] # Crop the image

  xaxis, yaxis = getAxes(img)
  print(f"X-axis: {xaxis}\nY-axis: {yaxis}")
  getLabels(img, xaxis, yaxis)
  
  # to maximize
  plt.get_current_fig_manager().full_screen_toggle()
  plt.show()


if __name__=='__main__':
  main()