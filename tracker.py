import numpy as np
import cv2
import glob

def getTemplate(img):
	bbox = [75,269,139,303]
	T = img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
	gray = cv2.cvtColor(T,cv2.COLOR_BGR2GRAY)

	return gray,bbox

def getWarp(img,tmp,P,gradx,grady):
	warp_im = np.zeros_like(tmp)
	warp_gradx = np.zeros_like(tmp)
	warp_grady = np.zeros_like(tmp)
	#print(1+P[0][0])
	Pm = np.array([[1+P[0][0],P[2][0],P[4][0]],[P[1][0],1+P[3][0],P[5][0]]])
	for x in range(tmp.shape[1]):
		for y in range(tmp.shape[0]):
			L = np.array([x,y,1]).T.reshape((3,1))
			W_L = np.int32(np.round(np.matmul(Pm,L)))
			warp_im[y,x] = img[W_L[1],W_L[0]]
			warp_gradx[y,x] = gradx[W_L[1],W_L[0]]
			warp_grady[y,x] = grady[W_L[1],W_L[0]]

	return warp_im,warp_gradx,warp_grady

def kidharGayaBe(gray,tmp,rect,pprev):
	img = gray
	P = pprev
	Ix = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
	Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
	thresh = 0.4
	err = 100
	while (err>thresh):
	#for i in range(200):
		wimg,wIx,wIy = getWarp(img,tmp,P,Ix,Iy)
		hess = np.zeros((6,6))
		ergrad = np.zeros((6,1))
		error = tmp - wimg
		for x in range(tmp.shape[1]):
			for y in range(tmp.shape[0]):
				jacobian = np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]])
				mgrad = np.array([wIx[y,x],wIy[y,x]])
				prod1 = np.matmul(mgrad,jacobian)
				#perror = tmp[y,x] - wimg[y,x]
				perror = error[y,x]
				prod1 = np.reshape(prod1,(1,6))
				perror = np.reshape(perror,(1,1))
				hess = hess + np.matmul(prod1.T,prod1)
				ergrad = ergrad + np.matmul(prod1.T,perror)

		del_p = np.matmul(np.linalg.inv(hess),ergrad)
		P = P + del_p
		pnorm = np.linalg.norm(del_p)
		err = pnorm

	return P

if __name__=="__main__":
	images = sorted(glob.glob('./Bolt2/img/*.jpg'))
	fname = './Bolt2/img/0001.jpg'
	img = cv2.imread(fname)
	gray_T,box = getTemplate(img)
	P = np.zeros((6,1))
	for im in images:
		IM = cv2.imread(im)
		GIM = cv2.cvtColor(IM,cv2.COLOR_BGR2GRAY)
		P = kidharGayaBe(GIM,gray_T,box,P)
		Pw = np.array([[1+P[0][0],P[2][0],P[4][0]],[P[1][0],1+P[3][0],P[5][0]]])
		box1 = np.array([box[1],box[0],1]).T.reshape((3,1))
		box4 = np.array([box[3],box[2],1]).T.reshape((3,1))
		wbox1 = np.matmul(Pw,box1)
		wbox4 = np.matmul(Pw,box4)
		
		IM = cv2.rectangle(IM,(wbox1[0],wbox1[1]),(wbox4[0],wbox4[1]),(255,0,0),3)
		cv2.imshow("Image",IM)
		cv2.waitKey(0)
	#img = cv2.rectangle(img, (269,75),(269+34,75+64), (255,0,0), 3)
	#cv2.imshow('Image',gray_T)
	#cv2.waitKey(0)