# trunc8 did this
import copy
import numpy as np


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
