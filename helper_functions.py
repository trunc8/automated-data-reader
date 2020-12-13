# trunc8 did this
import copy
import cv2
import numpy as np
from statistics import mean


def deep_copy_params(to_call):
  def f(*args, **kwargs):
    return to_call(*(copy.deepcopy(x) for x in args),
                   **{k: copy.deepcopy(v) for k, v in kwargs.items()})
  return f


def bestFitSlopeAndIntercept(xs,ys):
  xs = np.array(xs)
  ys = np.array(ys)

  m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
       ((mean(xs)*mean(xs)) - mean(xs*xs)))
  b = mean(ys) - m*mean(xs)
  
  return m, b


@deep_copy_params
def cropBlackBorder(image):
  y_nonzero, x_nonzero = np.nonzero(image)
  return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def findFirstNonZeroElement(lst):
  try:
    return [index for index, item in enumerate(lst) if item != 0][0]
  except:
    return -1


def findLastNonZeroElement(lst):
  try:
    return [index for index, item in enumerate(lst) if item != 0][-1]
  except:
    return -1


@deep_copy_params
def trimWhitespace(img):
  height, width = img.shape
  margin = 5
  img_copy = 255*(img < 128).astype(np.uint8) # To invert the text to white
  coords = cv2.findNonZero(img_copy) # Find all non-zero points (text)
  x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
  return img[max(0, y-margin):min(height, y+h+margin), 
             max(0, x-margin):min(width, x+w+margin)] # Crop the image