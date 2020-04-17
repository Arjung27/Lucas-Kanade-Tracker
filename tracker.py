import numpy as np
import cv2


if __name__=="__main__":
	fname = './Bolt2/img/0001.jpg'
	img = cv2.imread(fname)
	print(img.shape)
	img = cv2.rectangle(img, (269,75),(269+34,75+64), (255,0,0), 3)
	cv2.imshow('Image',img)
	cv2.waitKey(0)