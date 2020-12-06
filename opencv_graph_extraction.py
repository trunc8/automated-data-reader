# trunc8 did this
import argparse
import copy
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pytesseract


import imutils

def deep_copy_params(to_call):
  def f(*args, **kwargs):
    return to_call(*(copy.deepcopy(x) for x in args),
                   **{k: copy.deepcopy(v) for k, v in kwargs.items()})
  return f


def findLastNonZeroElement(lst):
  return [index for index, item in enumerate(lst) if item != 0][-1]


def findFirstNonZeroElement(lst):
  return [index for index, item in enumerate(lst) if item != 0][0]


@deep_copy_params
def crop(image):
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


@deep_copy_params
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
    'start': findFirstNonZeroElement(cropped_openedx[:,yaxis_col]),
    'end': findLastNonZeroElement(cropped_openedx[:,yaxis_col])
  }

  # processing opened y to get X-axis
  cropped_openedy = openedy[height//2:height, :] # bottom half of image
  cropped_openedy = 255*(cropped_openedy > 110).astype(np.uint8)
  sum_along_rows = np.sum(cropped_openedy, axis = 1)
  xaxis_row = np.argmax(sum_along_rows)
  xaxis = {
    'start': findFirstNonZeroElement(cropped_openedy[xaxis_row, :]),
    'end': findLastNonZeroElement(cropped_openedy[xaxis_row, :])
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


def eliminateYTitle(img):
  height, width = img.shape
  blur = cv2.GaussianBlur(img, (7,7), 0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  # Create rectangular structuring element and dilate
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,4))
  dilate = cv2.dilate(thresh, kernel, iterations=4)

  # Eliminate y title
  x_min = width-1
  cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if (x+w//1.5) > width/2:
      # cv2.rectangle(img, (x, y), (x + w, y + h), 0, 2)
      x_min = min(x_min, x)
  x_min = max(0, x_min-5)
  return img[:,x_min:] # crop the portion containing y-axis title


def eliminateYTicks(img):
  height, width = img.shape
  blur = cv2.GaussianBlur(img, (7,7), 0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  # Create rectangular structuring element and dilate
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,4))
  dilate = cv2.dilate(thresh, kernel, iterations=4)

  # Eliminate ticks
  x_max = 0
  cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if x-10 < width/2:
      # cv2.rectangle(img, (x, y), (x + w, y + h), 0, 2)
      x_max = max(x_max, x+w)
  x_max = min(width-1, x_max+5)
  return img[:,:x_max] # crop the portion containing y-axis tick marks


@deep_copy_params
def getYlabel(img, xaxis):
  label_images = []
  ylabels = []
  ypixels = []
  img = img[:,:max(0,xaxis['start']-5)]

  # Eliminate y title and tick marks
  img = eliminateYTitle(img)
  img = eliminateYTicks(img)
  
  # Obtain labels
  height, width = img.shape
  blur = cv2.GaussianBlur(img, (7,7), 0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
  dilate = cv2.dilate(thresh, kernel, iterations=4)

  cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    # cv2.rectangle(img, (x, y), (x + w, y + h), 0, 2)
    ypixels.append(y+h//2)
    margin = 5
    label_images.append(img[max(0,y-margin):min(height,y+h+margin), 
                            max(0,x-margin):min(width,x+w+margin)])
    
  ylabels = [None]*len(ypixels)
  for i,lab in enumerate(label_images):
    lab = cv2.resize(lab, None, fx=10.5, fy=10.5, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((1, 1), np.uint8)
    lab = cv2.dilate(lab, kernel, iterations=4)
    _, lab = cv2.threshold(lab, 200, 255, cv2.THRESH_BINARY)
    lab = cv2.GaussianBlur(lab,(5,5),0)
    text = pytesseract.image_to_string(lab, lang="eng", config="--psm 6 digits")
    # print(f"Label{i}: {text}")
    try:
      ylabels[i] = float(text)
    except:
      pass
    label_images[i] = lab
    # cv2.imshow(f'label{i}', lab)

  if (len(ylabels)-ylabels.count(None) < 3):
    # Not enough points to perform linear interpolation
    # We suspect that the numbers are rotated 90 deg CCW
    for i,lab in enumerate(label_images):
      lab = imutils.rotate(lab, -90)
      lab = crop(lab)
      text = pytesseract.image_to_string(lab, lang="eng", config="--psm 6 digits")
      # print(f"Label{i}: {text}")
      try:
        ylabels[i] = float(text)
      except:
        pass
      label_images[i] = lab
      # cv2.imshow(f'label{i}', lab)

  # cv2.imshow('thresh', thresh)
  # cv2.imshow('dilate', dilate)
  # cv2.imshow('image', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # # Rescale the image, if needed.
  # img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

  # # Apply dilation and erosion to remove some noise
  # kernel = np.ones((1, 1), np.uint8)
  # img = cv2.dilate(img, kernel, iterations=1)
  # img = cv2.erode(img, kernel, iterations=1)
  # # Apply blur to smooth out the edges
  # img = cv2.GaussianBlur(img, (5, 5), 0)

  # text = pytesseract.image_to_string(img, lang="eng", config="--psm 6 digits")
  # print(f'Y axis text\n{text}')

  plt.subplot(3,3,8),plt.imshow(img,cmap = 'gray')
  plt.title('Behind Y axis'), plt.xticks([]), plt.yticks([])
  zipped_y = zip(ypixels, ylabels)
  return zipped_y


def getLabels(img, xaxis, yaxis):
  zipped_y = getYlabel(img, xaxis)
  # behind_yaxis = cv2.resize(behind_yaxis, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
  # retval, behind_yaxis = cv2.threshold(behind_yaxis, 125, 255, cv2.THRESH_BINARY)
  # behind_yaxis = cv2.adaptiveThreshold(behind_yaxis, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
  # retval,behind_yaxis = cv2.threshold(behind_yaxis,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  # text = pytesseract.image_to_string(behind_yaxis, lang="eng", config="--psm 11")
  # print(f'Y axis text\n{text}')

  
  # retval, below_xaxis = cv2.threshold(below_xaxis, 125, 255, cv2.THRESH_BINARY)
  # below_xaxis = cv2.adaptiveThreshold(below_xaxis, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
  # retval,below_xaxis = cv2.threshold(below_xaxis,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  # text = pytesseract.image_to_string(below_xaxis, lang="eng", config="--psm 11")
  # print(f'X axis text\n{text}')

  below_xaxis = img[yaxis['end']:]
  plt.subplot(3,3,9),plt.imshow(below_xaxis,cmap = 'gray')
  plt.title('Below X axis'), plt.xticks([]), plt.yticks([])

  return zipped_y


def main():
  # Read args from terminal
  parser = argparse.ArgumentParser(description="Extract text from input image")
  parser.add_argument("-n", default="", help="image number")
  args = parser.parse_args()

  # Read file
  # filename = 'images/Line-Chart4.png'
  filename = f'images/Line-Chart{args.n}.png'
  img = cv2.imread(filename, 0) # read as grayscale image
  
  # Processing
  trimmed_img = trimWhitespace(img)
  xaxis, yaxis = getAxes(trimmed_img)
  zipped_y = getLabels(trimmed_img, xaxis, yaxis)

  print(f"X-axis: {xaxis}\nY-axis: {yaxis}")
  print("Y pixels:\n", list(zipped_y))
  
  # to maximize
  plt.get_current_fig_manager().full_screen_toggle()
  # plt.show()


if __name__=='__main__':
  main()