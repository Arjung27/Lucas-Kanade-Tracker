import numpy as np
import cv2
import glob
import os
import argparse
from copy import deepcopy

def getTemplate(img, i, key):

	if key.upper() == 'BOLT':
		bbox = [[75,269,139,303], [76, 225, 165, 259]] # bolt
	elif key.upper() == 'CAR':
		bbox = [[51,70,138,177]] # car
	elif key.upper() == 'BABY':
		bbox = [[83,160,148,216], [89, 188,154,253], [80,190,145,255]]  # baby image 1, 18, 28
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	template = img[bbox[i][0]:bbox[i][2],bbox[i][1]:bbox[i][3]]

	return template, bbox[i]


def getWarpy(img,tmp,P,rect,gradx,grady):

	Pm = np.array([[1+P[0][0],P[2][0],P[4][0]],[P[1][0],1+P[3][0],P[5][0]]],dtype='float32')
	Pm = cv2.invertAffineTransform(Pm)
	warp_img = cv2.warpAffine(img, Pm, (img.shape[1], img.shape[0]))
	warp_gradx = cv2.warpAffine(gradx, Pm, (gradx.shape[1], gradx.shape[0]))
	warp_grady = cv2.warpAffine(grady, Pm, (grady.shape[1], grady.shape[0]))

	return warp_img,warp_gradx,warp_grady


def kidharGayaBe(gray,tmp,rect,pprev, p_thresh):

	img = deepcopy(gray)
	img = np.asarray(img, dtype='uint8')
	gray = np.asarray(gray,dtype='float32')
	for i in range(len(tmp)):
		tmp[i] = np.asarray(tmp[i],dtype='float32')
	P = pprev
	P_i = P
	Ix = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
	Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
	thresh = 0.01
	err = [100]*len(tmp)
	ppnorm = 10
	count =0
	
	while (count <= 500):
		sigma = [0]*len(tmp) 		
		for k in range(len(tmp)):
			wimg,wIx,wIy = getWarpy(gray,tmp[k],P_i[k],rect,Ix,Iy)
			hess = np.zeros((6,6))
			ergrad = np.zeros((6,1))
			error = tmp[k]-wimg
			nerr = np.reshape(error,(-1,1))
			sigma[k] = np.sum(abs(nerr)) # Like Loss. Gives an idea of how good the warp is.
			std_sig = np.std(nerr)

			for i,x in enumerate(range(rect[k][1],rect[k][3]-1,1)):
				for j,y in enumerate(range(rect[k][0],rect[k][2]-1,1)):

					jacobian = np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]])
					mgrad = np.array([wIx[y,x],wIy[y,x]])
					prod1 = np.matmul(mgrad,jacobian)
					perror = tmp[k][y,x] - wimg[y,x]

					#This section contains the Huber Loss Implementation					
					t = perror**2
					if t <= std_sig**2:
						rho = 0.5*t
					else:
						rho = std_sig*np.sqrt(t) - 0.5*std_sig**2

					rho=1
					prod1 = np.reshape(prod1,(1,6))
					perror = np.reshape(perror,(1,1))
					hess = hess + rho*prod1.T*prod1
					ergrad = ergrad + rho*prod1.T*perror

			del_p = np.matmul(np.linalg.inv(hess),ergrad)
			P_i[k] = P_i[k] + del_p
			pnorm = np.linalg.norm(del_p)
			err[k] = pnorm
		count +=1

	idx = sigma.index(min(sigma))

	if not (np.linalg.norm(P_i) > p_thresh[idx]):
		
		if err[idx] > thresh:
			return P, None

		P[idx] = P_i[idx]

	return P, idx

def movingAverage(length, width, move_length, key):

	sum_move = (move_length+1)*move_length/2
	weights = [i/sum_move for i in range(1,move_length+1)]
	if key == 'baby':
		if len(length) >= move_length:
			mean_length = np.sum(weights*length[-move_length:])
			mean_width = np.sum(weights*width[-move_length:])
			length[len(length)-1] = mean_length
			width[len(width)-1] = mean_width

def kidharGayaBe_car(gray,tmp,rect,pprev):
	img = deepcopy(gray)
	img = np.asarray(img, dtype='uint8')
	gray = np.asarray(gray,dtype='float32')
	tmp = np.asarray(tmp,dtype='float32')
	P = pprev
	Ix = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
	Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
	thresh = 0.01
	err = 100
	ppnorm = 10
	count =0
	while (err>thresh):
	#for k in range(100):
		wimg,wIx,wIy = getWarpy(gray,tmp,P,rect,Ix,Iy)
		hess = np.zeros((6,6))
		ergrad = np.zeros((6,1))
		error = tmp-wimg
		nerr = np.reshape(error,(-1,1))
		sigma = np.std(nerr)
		for i,x in enumerate(range(rect[1],rect[3]-1,1)):
			for j,y in enumerate(range(rect[0],rect[2]-1,1)):
				jacobian = np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]])
				mgrad = np.array([wIx[y,x],wIy[y,x]])
				prod1 = np.matmul(mgrad,jacobian)
				perror = tmp[y,x] - wimg[y,x]
						
				'''
				#This section contains the Huber Loss Implementation
				
				t = perror**2
				if t<=sigma**2:
					rho=0.5*t
				else:
					rho = sigma*np.sqrt(t) - 0.5*sigma**2
				'''
				rho=1
				prod1 = np.reshape(prod1,(1,6))
				perror = np.reshape(perror,(1,1))
				hess = hess + rho*prod1.T*prod1
				ergrad = ergrad + rho*prod1.T*perror

		del_p = np.matmul(np.linalg.inv(hess),ergrad)
		P = P + del_p
		pnorm = np.linalg.norm(del_p)
		err = pnorm
		count +=1

		if np.linalg.norm(P)>185:
			return pprev
		if count>500:
			return pprev

	return P

def track_car(input_path, output_path):

	images = sorted(glob.glob(os.path.join(input_path, '*.jpg')), recursive=True)
	fname = input_path + '/0001.jpg'
	img = cv2.imread(fname)
	G = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray_T,box = getTemplate(img, 'car')
	P = np.zeros((6,1))

	h, w, _ = img.shape
	vidWriter = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))
	count = 0
	for im in images[0:]:
		IM = cv2.imread(im)
		GIM = cv2.cvtColor(IM,cv2.COLOR_BGR2GRAY)
		HGIM = cv2.equalizeHist(GIM)
		Pn = kidharGayaBe_car(GIM,G,box,P)
		P = Pn
		Pw = np.array([[1+Pn[0][0],Pn[2][0],Pn[4][0]],[Pn[1][0],1+Pn[3][0],Pn[5][0]]])
		box1 = np.array([box[1],box[0],1]).T.reshape((3,1))
		box4 = np.array([box[3],box[2],1]).T.reshape((3,1))
		wbox1 = np.matmul(Pw,box1)
		print(count)
		wbox4 = np.matmul(Pw,box4)
		
		IM = cv2.rectangle(IM,(wbox1[0],wbox1[1]),(wbox4[0],wbox4[1]),(255,0,0),3)
		vidWriter.write(IM)
		count +=1

	vidWriter.release()

def track_baby(input_path, output_path):

	length_list = np.array([])
	width_list = np.array([])

	images = sorted(glob.glob(os.path.join(input_path,'*.jpg')))
	fname = []

	fname.append(input_path + '/0001.jpg')  
	fname.append(input_path + '/0018.jpg')
	fname.append(input_path + '/0028.jpg')
	gray_temp = []
	img = []
	G = []
	box = []
	P = []
	p_thresh = [200]*len(fname)

	for i in range(len(fname)):
		img.append( cv2.imread(fname[i]))
		G.append(cv2.cvtColor(img[i],cv2.COLOR_BGR2GRAY))
		gray_temp_t, box_t = getTemplate(img[i], i, 'baby')
		gray_temp.append(gray_temp_t)
		box.append(box_t)
		P.append(np.zeros((6,1)))

	length = np.abs(box[0][2] - box[0][0])
	width = np.abs(box[0][3] - box[0][1])
	move_length = 5
	for i in range(move_length):
		length_list = np.append(length_list, length)
		width_list = np.append(width_list, width)
	
	h, w, _ = img[0].shape
	vidWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))
	count = 0
	Pn_prev = np.zeros((6,1))
	Pn = np.zeros((6,1))
	prev_idx = 0

	for im in images[0:]:

		IM = cv2.imread(im)
		GIM = cv2.cvtColor(IM,cv2.COLOR_BGR2GRAY)
		HGIM = cv2.equalizeHist(GIM)
		P = [np.zeros((6,1))]*3
		P, idx = kidharGayaBe(GIM,G,box,P, p_thresh)

		if idx is not None:
			Pn = P[idx]
			prev_idx = idx
		Pw = np.array([[1+Pn[0][0],Pn[2][0],Pn[4][0]],[Pn[1][0],1+Pn[3][0],Pn[5][0]]])
		box1 = np.array([box[prev_idx][1],box[prev_idx][0],1]).T.reshape((3,1))
		box4 = np.array([box[prev_idx][3],box[prev_idx][2],1]).T.reshape((3,1))
		wbox1 = np.matmul(Pw,box1)
		wbox4 = np.matmul(Pw,box4)
		
		centerx = int((wbox4[1] + wbox1[1])/2)
		centery = int((wbox4[0] + wbox1[0])/2)
		length = np.abs(wbox4[1] - wbox1[1])
		width = np.abs(wbox4[0] - wbox1[0])
		length_list = np.append(length_list, length)
		width_list = np.append(width_list, width)
		movingAverage(length_list, width_list, move_length, 'baby')
		newLeftx = int(centerx - length_list[-1]/2)
		newLefty = int(centery - width_list[-1]/2)
		newRightx = int(centerx + length_list[-1]/2)
		newRighty = int(centery + width_list[-1]/2)
		IM = cv2.rectangle(IM,(newLefty, newLeftx),(newRighty, newRightx),(255,0,0),3)
		vidWriter.write(IM)
		count +=1
		print(count)

	vidWriter.release()

def track_bolt(input_path, output_path):

	length_list = np.array([])
	width_list = np.array([])

	images = sorted(glob.glob(os.path.join(input_path,'*.jpg')))
	fname = []

	fname.append(input_path + '/0001.jpg')
	fname.append(input_path + '/0072.jpg')
	gray_temp = []
	img = []
	G = []
	box = []
	P = []
	p_thresh = [200]*len(fname)

	for i in range(len(fname)):
		img.append( cv2.imread(fname[i]))
		G.append(cv2.cvtColor(img[i],cv2.COLOR_BGR2GRAY))
		gray_temp_t, box_t = getTemplate(img[i], i, 'bolt')
		gray_temp.append(gray_temp_t)
		box.append(box_t)
		P.append(np.zeros((6,1)))

	length = np.abs(box[0][2] - box[0][0])
	width = np.abs(box[0][3] - box[0][1])
	move_length = 5
	for i in range(move_length):
		length_list = np.append(length_list, length)
		width_list = np.append(width_list, width)
	
	h, w, _ = img[0].shape
	vidWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))
	count = 0
	Pn_prev = np.zeros((6,1))
	Pn = np.zeros((6,1))
	prev_idx = 0

	for im in images[0:]:

		IM = cv2.imread(im)
		GIM = cv2.cvtColor(IM,cv2.COLOR_BGR2GRAY)
		P, idx = kidharGayaBe(GIM,G,box,P, p_thresh)

		if idx is not None:
			Pn = P[idx]
			prev_idx = idx
		Pw = np.array([[1+Pn[0][0],Pn[2][0],Pn[4][0]],[Pn[1][0],1+Pn[3][0],Pn[5][0]]])
		box1 = np.array([box[prev_idx][1],box[prev_idx][0],1]).T.reshape((3,1))
		box4 = np.array([box[prev_idx][3],box[prev_idx][2],1]).T.reshape((3,1))
		wbox1 = np.matmul(Pw,box1)
		wbox4 = np.matmul(Pw,box4)
		
		centerx = int((wbox4[1] + wbox1[1])/2)
		centery = int((wbox4[0] + wbox1[0])/2)
		length = np.abs(wbox4[1] - wbox1[1])
		width = np.abs(wbox4[0] - wbox1[0])
		length_list = np.append(length_list, length)
		width_list = np.append(width_list, width)
		movingAverage(length_list, width_list, move_length, 'baby')
		newLeftx = int(centerx - length_list[-1]/2)
		newLefty = int(centery - width_list[-1]/2)
		newRightx = int(centerx + length_list[-1]/2)
		newRighty = int(centery + width_list[-1]/2)
		IM = cv2.rectangle(IM,(newLefty, newLeftx),(newRighty, newRightx),(255,0,0),3)
		vidWriter.write(IM)
		count +=1
		print(count)
		# cv2.imwrite("./generation/car/" + str(count) + ".jpg",IM)

	vidWriter.release()

if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--object", default = 'car', help = "Name of the object to be tracked car/baby/bolt")
	parser.add_argument("--input", default = './Car4/img', help = "Path of the images")
	parser.add_argument("--output", default = './car_output.mp4', help = "Path of the output file")
	Flags = parser.parse_args()
	
	if Flags.object.upper() == 'CAR':
		track_car(Flags.input, Flags.output)
	
	elif Flags.object.upper() == 'BABY':
		track_baby(Flags.input, Flags.output)

	elif Flags.object.upper() == 'BOLT':
		track_bolt(Flags.input, Flags.output)
