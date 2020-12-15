# trunc8 did this
import collections
import copy
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

# Python scripts
import axis_processing, helper_functions


@helper_functions.deep_copy_params
def getYlabel(img, xaxis):
  label_images = []
  ylabels = []
  ypixels = []
  img = img[:,:max(0,xaxis['start']-5)]

  # Eliminate y title and tick marks
  img = axis_processing.eliminateYTitle(img)
  img = axis_processing.eliminateYTicks(img)
  
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
    
  ylabels = [float('inf')]*len(ypixels)
  plt.figure("y-label OCR")
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
    plt.subplot(3, len(ypixels)//3+1, i+1), plt.imshow(lab,cmap = 'gray')
    plt.title(f'label{i}'), plt.axis('off')
    # cv2.imshow(f'label{i}', lab)

  if (ylabels.count(float('inf')) > len(ylabels) - 3):
    # Not enough points to perform linear interpolation
    # We suspect that the numbers are rotated 90 deg CCW
    for i,lab in enumerate(label_images):
      lab = imutils.rotate(lab, -90)
      lab = helper_functions.cropBlackBorder(lab)
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

  plt.figure("Pre-OCR")
  plt.subplot(3,3,8),plt.imshow(img,cmap = 'gray')
  plt.title('Behind Y axis'), plt.xticks([]), plt.yticks([])

  zipped_y = zip(ypixels, ylabels)
  return list(zipped_y)


@helper_functions.deep_copy_params
def getXlabel(img, yaxis):
  label_images = []
  xlabels = []
  xpixels = []
  img = img[yaxis['end']+10:,:]

  # Eliminate y title and tick marks
  img = axis_processing.eliminateXTitle(img)
  img = axis_processing.eliminateXTicks(img)
  
  # Obtain labels
  height, width = img.shape
  blur = cv2.GaussianBlur(img, (7,7), 0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
  dilate = cv2.dilate(thresh, kernel, iterations=4)

  cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    # cv2.rectangle(img, (x, y), (x + w, y + h), 0, 2)
    xpixels.append(x+w//2)
    margin = 5
    label_images.append(img[max(0,y-margin):min(height,y+h+margin), 
                            max(0,x-margin):min(width,x+w+margin)])
    
  xlabels = [float('inf')]*len(xpixels)
  plt.figure("x-label OCR")
  for i,lab in enumerate(label_images):
    lab = cv2.resize(lab, None, fx=10.5, fy=10.5, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((1, 1), np.uint8)
    lab = cv2.dilate(lab, kernel, iterations=4)
    _, lab = cv2.threshold(lab, 200, 255, cv2.THRESH_BINARY)
    lab = cv2.GaussianBlur(lab,(5,5),0)
    text = pytesseract.image_to_string(lab, lang="eng", config="--psm 6 digits")
    # print(f"Label{i}: {text}")
    try:
      xlabels[i] = float(text)
    except:
      pass
    label_images[i] = lab
    plt.subplot(3, len(xpixels)//3+1, i+1),plt.imshow(lab,cmap = 'gray')
    plt.title(f'label{i}'), plt.axis('off')
    # cv2.imshow(f'label{i}', lab)

  # cv2.imshow('thresh', thresh)
  # cv2.imshow('dilate', dilate)
  # cv2.imshow('image', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  plt.figure("Pre-OCR")
  plt.subplot(3,3,9),plt.imshow(img,cmap = 'gray')
  plt.title('Below X axis'), plt.xticks([]), plt.yticks([])

  zipped_x = sorted(zip(xpixels, xlabels)) # index of both lists in ascending order of xpixels
  return list(zipped_x)


def cleanLabels(coord):
  # coord is composed of (pixels, labels)
  # Clean up labels data
  pixels, labels = map(list, zip(*coord))
  label_length = len(labels)
  cleaned_labels = copy.deepcopy(labels)

  # We have ordered sequence of labels.
  # Some values may have been detected incorrectly by OCR.
  # In order to find most reliable label, we use the idea of arithmetic
  # progression series to find the "common difference": a_{n}=a_{1}+(n-1)*d
  # We use that common difference to correct all other labels.
  # Formula used below: d = (a_{j}-a_{i})/(j-i)
  for i in range(label_length-1):
    if labels[i] != float('inf'): # All differences expected to be inf in this case. Discard.
      common_difference = [(labels[j]-labels[i])/(j-i) for j in range(i+1, label_length)]
      occurrence_frequences = collections.Counter(common_difference)
      mode, mode_freq = occurrence_frequences.most_common(1)[0]
      if (mode != float('inf') and mode_freq > 1):
        cleaned_labels = [labels[i]+(j-i)*mode for j in range(label_length)]
        break
  return list(zip(pixels, cleaned_labels))

def getLabels(img, xaxis, yaxis):
  xcoord = getXlabel(img, yaxis)
  ycoord = getYlabel(img, xaxis)
  cleaned_xcoord = cleanLabels(xcoord)
  cleaned_ycoord = cleanLabels(ycoord)
  return [cleaned_xcoord,cleaned_ycoord]