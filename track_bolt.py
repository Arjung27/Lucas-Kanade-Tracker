import numpy as np
import cv2
import glob
from copy import deepcopy

def getTemplate(img):
	bbox = [75,269,139,303]
	#bbox = [51,70,138,177]
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#img_eq = cv2.equalizeHist(gray)
	template = gray[bbox[0]:bbox[2],bbox[1]:bbox[3]]
	

	return template, bbox

def getTemp(img):
	bbox = [75,269,139,303]
	#bbox = [51,70,138,177]
	T = img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
	#gray = cv2.cvtColor(T,cv2.COLOR_BGR2GRAY)

	return T

def getWarpy(img,tmp,P,rect,gradx,grady):
	Pm = np.array([[1+P[0][0],P[2][0],P[4][0]],[P[1][0],1+P[3][0],P[5][0]]],dtype='float32')
	Pm = cv2.invertAffineTransform(Pm)
	warp_img = cv2.warpAffine(img, Pm, (img.shape[1], img.shape[0]))
	# cv2.imshow('warp',warp_img)
	# cv2.waitKey(0)
	#warp_gradx = cv2.Sobel(warp_img,cv2.CV_64F,1,0,ksize=3)
	#warp_grady = cv2.Sobel(warp_img,cv2.CV_64F,0,1,ksize=3)
	warp_gradx = cv2.warpAffine(gradx, Pm, (gradx.shape[1], gradx.shape[0]))
	warp_grady = cv2.warpAffine(grady, Pm, (grady.shape[1], grady.shape[0]))
	#warp_im = getTemp(warp_img)
	#cv2.imshow('warp',warp_img)
	#cv2.waitKey(0)
	return warp_img,warp_gradx,warp_grady


def kidharGayaBe(gray,tmp,rect,pprev):
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
		# print(count)
		if np.linalg.norm(P)>200:
			return pprev
		#if count>500:
		#	return pprev
		#print(pnorm)

	return P


if __name__=="__main__":
	images = sorted(glob.glob('./Bolt2/img/*.jpg'))
	fname = './Bolt2/img/0001.jpg'
	#images = sorted(glob.glob('./Car4/img/*.jpg'))
	#fname = './Car4/img/0001.jpg'
	img = cv2.imread(fname)
	G = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# cv2.imshow("raw_img", img)
	# cv2.waitKey(0)
	gray_T,box = getTemplate(img)
	# cv2.imshow("hist eq", gray_T)
	# cv2.waitKey(0)
	# gray_T_eq = 
	# IM = cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),3)
	# cv2.imshow("s", IM)
	# cv2.waitKey(0)
	P = np.zeros((6,1))
	vidWriter = cv2.VideoWriter("bhaag_bolt.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 24, (480,270))
	count = 0
	for im in images[1:]:
		IM = cv2.imread(im)
		GIM = cv2.cvtColor(IM,cv2.COLOR_BGR2GRAY)
		# cv2.imshow("gray", GIM)
		# cv2.waitKey(0)
		HGIM = cv2.equalizeHist(GIM)
		# cv2.imshow("hist gray", HGIM)
		# cv2.waitKey(0)
		P = np.zeros((6,1))
		Pn = kidharGayaBe(GIM,G,box,P)
		#P = Pn
		Pw = np.array([[1+Pn[0][0],Pn[2][0],Pn[4][0]],[Pn[1][0],1+Pn[3][0],Pn[5][0]]])
		box1 = np.array([box[1],box[0],1]).T.reshape((3,1))
		box4 = np.array([box[3],box[2],1]).T.reshape((3,1))
		wbox1 = np.matmul(Pw,box1)
		#print(np.linalg.norm(P))
		print(count)
		wbox4 = np.matmul(Pw,box4)
		
		IM = cv2.rectangle(IM,(wbox1[0],wbox1[1]),(wbox4[0],wbox4[1]),(255,0,0),3)
		vidWriter.write(IM)
		count +=1
		cv2.imshow("Image",IM)
		cv2.waitKey(10)
		# print('Done')
	vidWriter.release()
	#img = cv2.rectangle(img, (269,75),(269+34,75+64), (255,0,0), 3)
	#cv2.imshow('Image',gray_T)
	#cv2.waitKey(0)