import cv2
import os
import glob
import argparse
import numpy as np

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--input", default = './list_baby.txt', help = "Path of the images")
	parser.add_argument("--output", default = './baby_output.mp4', help = "Path of the output file")
	Flags = parser.parse_args()

	# files = glob.glob(os.path.join(Flags.input, '*.jpg'), recursive=True)
	files = open(Flags.input, 'r')
	lines = files.readlines()
	image0 = cv2.imread(lines[0].rstrip())
	h, w, _ = image0.shape
	vidWriter = cv2.VideoWriter(Flags.output,cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))

	for line in lines:
		image = line.rstrip()
		img = cv2.imread(image)
		vidWriter.write(img)

	vidWriter.release()
