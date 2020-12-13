# trunc8 did this
import copy
import collections

# Python scripts
import helper_functions

def mapPixelToCoordinate(coord):
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
  
  # Fitting line through the (pixels, cleaned_labels) data
  m, b = helper_functions.bestFitSlopeAndIntercept(pixels, cleaned_labels)
  return m, b