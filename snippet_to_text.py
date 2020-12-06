# trunc8 did this
from io import StringIO
from PIL import Image

import argparse
import pytesseract

# Read args from terminal
parser = argparse.ArgumentParser(description="Extract text from input image")
parser.add_argument("-i", "--image", default="images/test.png", help="image file path")
args = parser.parse_args()

# Perform OCR using tesseract
im = Image.open(args.image)
text = pytesseract.image_to_string(im, lang="eng")

# Post-processing to form paragraphs of continuous text
processed_text = "" # Empty string
for line in StringIO(text):
  if line != "\n":
    line = line.rstrip('\n')
    if line[-1] != "-":
      line += " "
  else:
    line += "\n"
  processed_text += line

print(processed_text)
