# trunc8 did this
import argparse
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np

# Python scripts
import coordinate_mapper, helper_functions, opencv_graph_extraction, label_reader

def main():
  logging.info("The program is starting...")
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
  trimmed_img = helper_functions.trimWhitespace(img)

  logging.info("Getting X and Y axes...")
  xaxis, yaxis = opencv_graph_extraction.getAxes(trimmed_img)
  logging.info("Done")

  logging.info("Extracting and reading X and Y labels...")
  xcoord, ycoord = label_reader.getLabels(trimmed_img, xaxis, yaxis)
  logging.info("Done")

  # Debug data
  logging.debug("X axis pixels:")
  logging.debug(xaxis)
  logging.debug("")
  logging.debug("Y axis pixels:")
  logging.debug(yaxis)
  logging.debug("")
  logging.debug("X coordinates:")
  logging.debug(xcoord)
  logging.debug("")
  logging.debug("Y coordinates:")
  logging.debug(ycoord)
  logging.debug("")

  logging.info("Mapping pixels to coordinates...")
  m_y, b_y = coordinate_mapper.mapPixelToCoordinate(ycoord)
  m_x, b_x = coordinate_mapper.mapPixelToCoordinate(xcoord)
  logging.info("Done")

  logging.info("Extracting points in the graph plot...")
  opencv_graph_extraction.extractPlot(trimmed_img, xaxis, yaxis, m_x, b_x, m_y, b_y)
  logging.info("Done")

  # to maximize
  # plt.get_current_fig_manager().full_screen_toggle()
  # plt.show()
  logging.info("The program ended successfully!")


if __name__=='__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')
  ## Below is a toggle switch for logging messages
  # logging.disable(sys.maxsize)
  try:
    main()
  except Exception as e:
    logging.debug(f"Exception occurred: {e}")
    logging.info("Sorry! The program is unable to process this graph as of now")