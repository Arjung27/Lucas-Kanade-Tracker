import numpy as np
import cv2
import glob
from copy import deepcopy

def getTemplate_baby(img, i):
	#bbox = [75,269,139,303] # bolt
	# bbox = [51,70,138,177] # car
	# bbox = [83,160,148,216] # baby
	bbox = [[83,160,148,216], [89, 188,154,253], [80,190,145,255]]  # baby image 1, 18, 28
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# img_eq = cv2.equalizeHist(gray)
	template = img[bbox[i][0]:bbox[i][2],bbox[i][1]:bbox[i][3]]

	return template, bbox[i]

def getTemplate_bolt(img, i):
	bbox = [[75,269,139,303]] # bolt
	# bbox = [51,70,138,177] # car
	# bbox = [83,160,148,216] # baby
	# bbox = [[83,160,148,216], [89, 188,154,253], [80,190,145,255]]  # baby image 1, 18, 28
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# img_eq = cv2.equalizeHist(gray)
	template = img[bbox[i][0]:bbox[i][2],bbox[i][1]:bbox[i][3]]

	return template, bbox[i]


def getTemp(img):
	#bbox = [75,269,139,303] # bolt
	# bbox = [51,70,138,177] # car
	# bbox = [83,160,148,216] # baby
	bbox = [89, 188,154,253] # baby image 18
	T = img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
	#gray = cv2.cvtColor(T,cv2.COLOR_BGR2GRAY)

	return T
'''
def getWarp(img,tmp,P,rect,gradx,grady):
	warp_im = np.zeros_like(tmp)
	warp_gradx = np.zeros_like(tmp)
	warp_grady = np.zeros_like(tmp)
	#print(1+P[0][0])
	Pm = np.array([[1+P[0][0],P[2][0],P[4][0]],[P[1][0],1+P[3][0],P[5][0]]])
	for i,x in enumerate(range(rect[1],rect[3]-1,1)):
		for j,y in enumerate(range(rect[0],rect[2]-1,1)):
			L = np.array([x,y,1]).T.reshape((3,1))
			W_L = np.int64(np.round(np.matmul(Pm,L)))
			warp_im[j,i] = img[W_L[1],W_L[0]]
			warp_gradx[j,i] = gradx[W_L[1],W_L[0]]
			warp_grady[j,i] = grady[W_L[1],W_L[0]]

	#cv2.imshow('Inter',warp_im)
	#cv2.waitKey(0)
	return warp_im,warp_gradx,warp_grady
'''
def getWarpy(img,tmp,P,rect,gradx,grady):
	Pm = np.array([[1+P[0][0],P[2][0],P[4][0]],[P[1][0],1+P[3][0],P[5][0]]],dtype='float32')
	Pm = cv2.invertAffineTransform(Pm)
	# cv2.imshow('warp',img.astype(np.uint8))
	# cv2.waitKey(0)
	warp_img = cv2.warpAffine(img, Pm, (img.shape[1], img.shape[0]))
	# cv2.imshow('warp',warp_img.astype(np.uint8))
	# cv2.waitKey(0)
	#warp_gradx = cv2.Sobel(warp_img,cv2.CV_64F,1,0,ksize=3)
	#warp_grady = cv2.Sobel(warp_img,cv2.CV_64F,0,1,ksize=3)
	warp_gradx = cv2.warpAffine(gradx, Pm, (gradx.shape[1], gradx.shape[0]))
	warp_grady = cv2.warpAffine(grady, Pm, (grady.shape[1], grady.shape[0]))
	#warp_im = getTemp(warp_img)
	#cv2.imshow('warp',warp_img)
	#cv2.waitKey(0)
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
			for i,x in enumerate(range(rect[k][1],rect[k][3]-1,1)):
				for j,y in enumerate(range(rect[k][0],rect[k][2]-1,1)):
					jacobian = np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]])
					mgrad = np.array([wIx[y,x],wIy[y,x]])
					prod1 = np.matmul(mgrad,jacobian)
					perror = tmp[k][y,x] - wimg[y,x]
					#perror = error[y,x]
					
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

		# if count>500:
		# 	print("Couldn't converge :( ")	
		# 	# couldn't converge, so return the P corresponding to least sigma and its id to draw the bounding box
		# 	idx = None
		# 	return P, idx 

	# return the P which has converged along with its id to draw the corresponding bounding box
	# idx = err.index(min(err))
	# if not (np.linalg.norm(P_i) > p_thresh[idx]):
	# 	P[idx] = P_i[idx]

	idx = sigma.index(min(sigma))

	if not (np.linalg.norm(P_i) > p_thresh[idx]):
		
		if err[idx] > thresh:
			return P, None

		P[idx] = P_i[idx]

	return P, idx

def movingAverage(length, width, key):

	weights = np.array([0.2, 0.3, 0.5])
	if key == 'baby':
		if len(length) >= 3:
			mean_length = np.sum(weights*length[-3:])
			mean_width = np.sum(weights*width[-3:])
			length[len(length)-1] = mean_length
			width[len(width)-1] = mean_width

	# return length, width


if __name__=="__main__":
	
	#### BABY
	images = sorted(glob.glob('./DragonBaby/DragonBaby/img/*.jpg'))
	fname = []

	fname.append('./DragonBaby/DragonBaby/img/0001.jpg')  
	fname.append('./DragonBaby/DragonBaby/img/0018.jpg')
	fname.append('./DragonBaby/DragonBaby/img/0028.jpg')
	gray_temp = []
	img = []
	G = []
	box = []
	P = []
	length_list = np.array([])
	width_list = np.array([])
	p_thresh = [200]*len(fname)
	# cv2.imshow("raw_img", img)
	# cv2.waitKey(0)
	for i in range(len(fname)):
		img.append( cv2.imread(fname[i]))
		G.append(cv2.cvtColor(img[i],cv2.COLOR_BGR2GRAY))
		gray_temp_t, box_t = getTemplate_baby(img[i], i)
		gray_temp.append(gray_temp_t)
		box.append(box_t)
		P.append(np.zeros((6,1)))
	####


	# #### BOLT
	# images = sorted(glob.glob('./Bolt2/img/*.jpg'))
	# fname = []

	# fname.append('./Bolt2/img/0001.jpg')  
	# gray_temp = []
	# img = []
	# G = []
	# box = []
	# P = []
	# p_thresh = [200]*len(fname)
	# # cv2.imshow("raw_img", img)
	# # cv2.waitKey(0)
	# for i in range(len(fname)):
	# 	img.append( cv2.imread(fname[i]))
	# 	G.append(cv2.cvtColor(img[i],cv2.COLOR_BGR2GRAY))
	# 	gray_temp_t, box_t = getTemplate_bolt(img[i], i)
	# 	gray_temp.append(gray_temp_t)
	# 	box.append(box_t)
	# 	P.append(np.zeros((6,1)))
	# ####


	# #### CAR
	# images = sorted(glob.glob('./Car4/img/*.jpg'))
	# fname = []

	# fname.append('./Car4/img/0001.jpg')  
	# gray_temp = []
	# img = []
	# G = []
	# box = []
	# P = []
	# # cv2.imshow("raw_img", img)
	# # cv2.waitKey(0)
	# p_thresh = [200]*len(fname)
	# for i in range(len(fname)):
	# 	img.append( cv2.imread(fname[i]))
	# 	G.append(cv2.cvtColor(img[i],cv2.COLOR_BGR2GRAY))
	# 	gray_temp_t, box_t = getTemplate(img[i], i)
	# 	gray_temp.append(gray_temp_t)
	# 	box.append(box_t)
	# 	P.append(np.zeros((6,1)))
	# ####

	length = np.abs(box[0][2] - box[0][0])
	width = np.abs(box[0][3] - box[0][1])

	for i in range(3):
		length_list = np.append(length_list, length)
		width_list = np.append(width_list, width)
	
	vidWriter = cv2.VideoWriter("video_baby.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 24, (640,360))
	count = 0
	Pn_prev = np.zeros((6,1))
	Pn = np.zeros((6,1))
	prev_idx = 0
	for im in images[0:]:
		# if count < 72:
		# 	count+=1
		# 	continue
		IM = cv2.imread(im)
		GIM = cv2.cvtColor(IM,cv2.COLOR_BGR2GRAY)
		# cv2.imshow("gray", GIM)
		# cv2.waitKey(0)
		# HGIM = cv2.equalizeHist(GIM)
		# cv2.imshow("hist gray", HGIM)
		# cv2.waitKey(0)
		P = [np.zeros((6,1))]*3
		P, idx = kidharGayaBe(GIM,G,box,P, p_thresh)
		print("idx is ", idx)
		if idx is not None:
			Pn = P[idx]
			prev_idx = idx
		Pw = np.array([[1+Pn[0][0],Pn[2][0],Pn[4][0]],[Pn[1][0],1+Pn[3][0],Pn[5][0]]])
		box1 = np.array([box[prev_idx][1],box[prev_idx][0],1]).T.reshape((3,1))
		box4 = np.array([box[prev_idx][3],box[prev_idx][2],1]).T.reshape((3,1))
		wbox1 = np.matmul(Pw,box1)
		# print(count)
		wbox4 = np.matmul(Pw,box4)
		
		centerx = int((wbox4[1] + wbox1[1])/2)
		centery = int((wbox4[0] + wbox1[0])/2)
		length = np.abs(wbox4[1] - wbox1[1])
		width = np.abs(wbox4[0] - wbox1[0])
		length_list = np.append(length_list, length)
		width_list = np.append(width_list, width)
		movingAverage(length_list, width_list, 'baby')
		newLeftx = int(centerx - length_list[-1]/2)
		newLefty = int(centery - width_list[-1]/2)
		newRightx = int(centerx + length_list[-1]/2)
		newRighty = int(centery + width_list[-1]/2)
		print(length_list, width_list)

		IM = cv2.rectangle(IM,(newLefty, newLeftx),(newRighty, newRightx),(255,0,0),3)
		vidWriter.write(IM)
		count +=1
		cv2.imwrite("./generation/baby_new_/" + str(count) + ".jpg",IM)
	vidWriter.release()
