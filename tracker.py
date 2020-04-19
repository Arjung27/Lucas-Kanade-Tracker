import numpy as np
import cv2
import glob

def getTemplate(img):
	#bbox = [75,269,139,303]
	bbox = [51,70,138,177]
	T = img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
	gray = cv2.cvtColor(T,cv2.COLOR_BGR2GRAY)

	return gray,bbox

def getTemp(img):
	#bbox = [75,269,139,303]
	bbox = [51,70,138,177]
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
	Pm = np.array([[1+P[0][0],P[2][0],P[4][0]],[P[1][0],1+P[3][0],P[5][0]]])
	warp_img = cv2.warpAffine(img, Pm, (img.shape[1], img.shape[0]))
	#cv2.imshow('warp',warp_img)
	#cv2.waitKey(0)
	warp_gradx = cv2.Sobel(warp_img,cv2.CV_64F,1,0,ksize=3)
	warp_grady = cv2.Sobel(warp_img,cv2.CV_64F,0,1,ksize=3)
	warp_im = getTemp(warp_img)

	return warp_im,warp_gradx,warp_grady


def kidharGayaBe(gray,tmp,rect,pprev):
	img = gray
	P = pprev
	Ix = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
	Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
	thresh = 0.05
	err = 100
	ppnorm = 10
	while (err>thresh):
	#for k in range(100):
		wimg,wIx,wIy = getWarpy(img,tmp,P,rect,Ix,Iy)
		hess = np.zeros((6,6))
		ergrad = np.zeros((6,1))
		error = tmp - wimg
		nerr = np.reshape(error,(-1,1))
		sigma = np.std(nerr)
		for i,x in enumerate(range(rect[1],rect[3]-1,1)):
			for j,y in enumerate(range(rect[0],rect[2]-1,1)):
				jacobian = np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]])
				mgrad = np.array([wIx[y,x],wIy[y,x]])
				prod1 = np.matmul(mgrad,jacobian)
				#perror = tmp[y,x] - wimg[y,x]
				perror = error[j,i]
				prod1 = np.reshape(prod1,(1,6))
				perror = np.reshape(perror,(1,1))
				hess = hess + prod1.T*prod1
				ergrad = ergrad + prod1.T*perror

		del_p = np.matmul(np.linalg.inv(hess),ergrad)
		P = P + del_p
		pnorm = np.linalg.norm(del_p)
		err = pnorm
		
		print(pnorm)

	return P

if __name__=="__main__":
	#images = sorted(glob.glob('./Bolt2/img/*.jpg'))
	#fname = './Bolt2/img/0001.jpg'
	images = sorted(glob.glob('./Car4/img/*.jpg'))
	fname = './Car4/img/0001.jpg'
	img = cv2.imread(fname)
	gray_T,box = getTemplate(img)
	P = np.zeros((6,1))
	
	for im in images[1:]:
		IM = cv2.imread(im)
		GIM = cv2.cvtColor(IM,cv2.COLOR_BGR2GRAY)
		Pn = kidharGayaBe(GIM,gray_T,box,P)
		P = Pn
		Pw = np.array([[1+Pn[0][0],Pn[2][0],Pn[4][0]],[Pn[1][0],1+Pn[3][0],Pn[5][0]]])
		box1 = np.array([box[1],box[0],1]).T.reshape((3,1))
		box4 = np.array([box[3],box[2],1]).T.reshape((3,1))
		wbox1 = np.matmul(Pw,box1)
		print(wbox1)
		wbox4 = np.matmul(Pw,box4)
		
		IM = cv2.rectangle(IM,(wbox1[0],wbox1[1]),(wbox4[0],wbox4[1]),(255,0,0),3)
		cv2.imshow("Image",IM)
		cv2.waitKey(0)
		print('Done')
	
	#img = cv2.rectangle(img, (269,75),(269+34,75+64), (255,0,0), 3)
	#cv2.imshow('Image',gray_T)
	#cv2.waitKey(0)