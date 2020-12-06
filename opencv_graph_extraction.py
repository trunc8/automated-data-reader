# trunc8 did this
import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def find_last_non_zero_element(lst):
  return [index for index, item in enumerate(lst) if item != 0][-1]

def find_first_non_zero_element(lst):
  return [index for index, item in enumerate(lst) if item != 0][0]

def main():
  # Read args from terminal
  parser = argparse.ArgumentParser(description="Extract text from input image")
  parser.add_argument("-n", default="", help="image number")
  args = parser.parse_args()
  filename = f'images/Line-Chart{args.n}.png'

  img = cv2.imread(filename)
  height = img.shape[0]
  width = img.shape[1]
  # cv2.imshow('img.jpg',img)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  # cv2.imshow('gray.jpg',gray)

  # Trim whitespace
  margin = 10
  gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
  coords = cv2.findNonZero(gray) # Find all non-zero points (text)
  x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
  img = img[max(0, y-margin):min(height, y+h+margin), 
            max(0, x-margin):min(width, x+w+margin)] # Crop the image
  # cv2.imshow("cropped.jpg", img)
  plt.subplot(3,3,1),plt.imshow(img,cmap = 'gray')
  plt.title('Cropped'), plt.xticks([]), plt.yticks([])

  # Horizontal and vertical edges
  sobelx = np.uint8(np.absolute(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)))
  sobely = np.uint8(np.absolute(cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)))
  # cv2.imshow('sobelx.jpg', sobelx)
  # cv2.imshow('sobely.jpg', sobely)
  plt.subplot(3,3,2),plt.imshow(sobelx,cmap = 'gray')
  plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
  plt.subplot(3,3,3),plt.imshow(sobely,cmap = 'gray')
  plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

  kernelx = np.ones((img.shape[1]//10, 1), np.uint8)
  openedx = cv2.morphologyEx(sobelx, cv2.MORPH_OPEN, kernelx)
  kernely = np.ones((1, img.shape[0]//10), np.uint8)
  openedy = cv2.morphologyEx(sobely, cv2.MORPH_OPEN, kernely)

  plt.subplot(3,3,4),plt.imshow(openedx,cmap = 'gray')
  plt.title('Opened Sobel X'), plt.xticks([]), plt.yticks([])
  plt.subplot(3,3,5),plt.imshow(openedy,cmap = 'gray')
  plt.title('Opened Sobel Y'), plt.xticks([]), plt.yticks([])

  ## processing opened x to get Y-axis
  openedx = cv2.cvtColor(openedx,cv2.COLOR_BGR2GRAY)
  cropped_img = openedx[:,0:openedx.shape[1]//2]
  openedx = cropped_img
  mask = openedx > 110
  openedx[mask] = 255
  openedx[~mask] = 0
  plt.subplot(3,3,6),plt.imshow(openedx,cmap = 'gray')
  plt.title('Y axis'), plt.xticks([]), plt.yticks([])

  sum_along_columns = np.sum(openedx, axis = 0)
  yaxis_col = np.argmax(sum_along_columns)
  yaxis = {
    'start': find_first_non_zero_element(openedx[:,yaxis_col]),
    'end': find_last_non_zero_element(openedx[:,yaxis_col])
  }
  print(yaxis)

  ## processing opened y to get X-axis
  openedy = cv2.cvtColor(openedy,cv2.COLOR_BGR2GRAY)
  cropped_img = openedy[openedy.shape[0]//2:openedy.shape[0], :]
  openedy = cropped_img
  mask = openedy > 110
  openedy[mask] = 255
  openedy[~mask] = 0
  plt.subplot(3,3,7),plt.imshow(openedy,cmap = 'gray')
  plt.title('X axis'), plt.xticks([]), plt.yticks([])

  sum_along_rows = np.sum(openedy, axis = 1)
  xaxis_col = np.argmax(sum_along_rows)
  xaxis = {
    'start': find_first_non_zero_element(openedy[xaxis_col, :]),
    'end': find_last_non_zero_element(openedy[xaxis_col, :])
  }
  print(xaxis)

  # to maximize
  mng = plt.get_current_fig_manager()
  mng.full_screen_toggle()
  plt.show()
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()


if __name__=='__main__':
  main()