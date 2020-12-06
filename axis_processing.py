# trunc8 did this
import cv2
import numpy as np


def eliminateXTitle(img):
  height, width = img.shape
  blur = cv2.GaussianBlur(img, (7,7), 0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  # Create rectangular structuring element and dilate
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,1))
  dilate = cv2.dilate(thresh, kernel, iterations=4)

  # Eliminate x title
  y_max = height
  cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  # Check if there is any x title
  # findContours goes from bottom to up, so title would be the first element in cnts
  if (len(cnts)>1):
  	_,y_title,_,h_title = cv2.boundingRect(cnts[0]) # suspected to be x title
  	x,y,w,h = cv2.boundingRect(cnts[1])
  	if (y_title > y+h):
  		# Our suspicion is true. cnts[0] is the x title
  		y_max = y+h
  
  y_max = min(height, y_max+5)
  return img[:y_max,:] # crop the portion containing x-axis title


def eliminateXTicks(img):
  height, width = img.shape
  blur = cv2.GaussianBlur(img, (7,7), 0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  # Create rectangular structuring element and dilate
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,4))
  dilate = cv2.dilate(thresh, kernel, iterations=4)

  # Eliminate ticks
  y_min = height
  cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if y+w-5 > height/2:
      # cv2.rectangle(img, (x, y), (x + w, y + h), 0, 2)
      y_min = min(y_min, y)
  y_min = max(0, y_min-5)
  return img[y_min:,:] # crop the portion containing x-axis tick marks


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
    if (x+w/1.5) > width/2:
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