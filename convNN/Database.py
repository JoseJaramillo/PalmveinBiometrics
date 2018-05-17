"""
Artificial Intelligence & Robotics
Biometric systems
Palmvein recognition / Deep Neural Networks
Jose Vicente Jaramillo
Extract library
"""

#from PIL import Image, ImageFilter, ImageOps
#import cv2
from matplotlib import pyplot as plt
import cv2
import numpy as np

def autocontrast(image):



    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    
    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
#    cv2.imshow('Increased contrast', img2)
#    #cv2.imwrite('sunset_modified.jpg', img2)
#    
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    cv2.imshow('Increased contrast', image)
#    #cv2.imwrite('sunset_modified.jpg', img2)
#    
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    return img2
def extract(directory):
    
##############################################################################
#Resizing the image to 64x48 an accuracy of 44.5% was reached (Pooling=true)
#Resizing the image to 128x96 an accuracy of 33% was reached (Pooling=true)
##############################################################################
    resizeX=64
    resizeY=48
    #Any feature extraction technique should be implemented here.
    
    #Will read an rgb file, grayscale it, and resize it to 128x96pix 
    #and return a 1D list of gray levels
    #Images are RGB, convert('L') will transform them to grayscale
    #images are too big, resize has been applyed 128x96
    
#    aa=Image.open(directory).resize((128,96)).convert('L')
#    ab= list(aa.getdata())
#    return ab
    
    #edgedettection?
#    aa=Image.open(directory).convert('L')
#    bb=cv2.imread(directory,cv2.IMREAD_GRAYSCALE)
    bb=cv2.imread(directory)
    bb=autocontrast(bb)
    bb = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY)
#    aa=equalize(aa)
#    aa.show()
    
    #thresholding
    #cont = Image.fromarray(bbblur)
#    cont=ImageOps.autocontrast(aa)
#    contcv=np.array(cont) 
    
    
    bbblur=cv2.medianBlur(bb,25)

    #edges image
    edgeImage=np.array(bbblur) 
    edgeImage = edgeImage[:, ::1].copy() 
    edgeImage= cv2.Canny(bbblur,3,20) #3,7
    edgeImage = cv2.blur(edgeImage,(5,5))
    #Thick the lines
    
    dilatated = cv2.dilate(edgeImage,cv2.getStructuringElement(cv2.MORPH_RECT,(14,14)),iterations = 1)
    
#    aa=aa.filter(ImageFilter.BLUR)
#    plt.subplot(121),plt.imshow(aa,cmap = 'gray')
#    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#    aa=ImageOps.autocontrast(aa, cutoff=2, ignore=None)
#    aa.show()
#    aa=aa.resize((128,96))
#    plt.subplot(121),plt.imshow(aa,cmap = 'gray')
#    aa=aa.filter(ImageFilter.EDGE_ENHANCE_MORE)
#    aa=aa.filter(ImageFilter.FIND_EDGES)
#    aa=ImageOps.autocontrast(aa, cutoff=30, ignore=None)
#    openImage=np.array(aa) 
#    openImage = openImage[:, ::1].copy() 
    #Blur
    openImage = cv2.blur(dilatated,(50,50))
    
#    plt.subplot(121),plt.imshow(aa,cmap = 'gray')
#    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#    plt.subplot(122),plt.imshow(openImage,cmap = 'gray')
#    plt.title('smoth Image'), plt.xticks([]), plt.yticks([])



###################### PLOTS ######################

#    plt.subplot(321),plt.imshow(bb,cmap = 'gray')
#    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
##    plt.subplot(322),plt.imshow(cont,cmap = 'gray')
##    plt.title('contrast Image'), plt.xticks([]), plt.yticks([])
#    plt.subplot(323),plt.imshow(bbblur,cmap = 'gray')
#    plt.title('smoth Image'), plt.xticks([]), plt.yticks([])
#    plt.subplot(324),plt.imshow(edgeImage,cmap = 'gray')
#    plt.title('edged Image'), plt.xticks([]), plt.yticks([])
#    plt.subplot(325),plt.imshow(dilatated,cmap = 'gray')
#    plt.title('thick edged Image'), plt.xticks([]), plt.yticks([])
#    plt.subplot(326),plt.imshow(openImage,cmap = 'gray')
#    plt.title('Blured edged Image'), plt.xticks([]), plt.yticks([])
    
###################### !PLOTS ######################

###################### SIFT ######################
#####Resize
    
    resized_image = cv2.resize(bb, (resizeX, resizeY)) #128,96
    resized_image1 = cv2.resize(bbblur, (resizeX, resizeY)) #128,96
#    plt.subplot(326),plt.imshow(resized_image,cmap = 'gray')
#    plt.title('resized Image'), plt.xticks([]), plt.yticks([])
#    sift = cv2.xfeatures2d.SIFT_create()
#    kp, des = sift.detectAndCompute(resized_image,None)


#    pil.show()
#    plt.imshow(resized_image,cmap = 'gray')
    resized_image1=resized_image1.flatten()
#    resized_image=resized_image.transpose()
    resized_image1=resized_image1.reshape(1,resizeX*resizeY)
    resized_image=resized_image.flatten()
#    resized_image=resized_image.transpose()
    resized_image=resized_image.reshape(1,resizeX*resizeY)
    #heavy processed image
    return resized_image1   
    #resized with autocontrast
#    return resized_image


#aa=Image.open('Database/Palm/o_001/Left/Series_1/P_o001_L_S1_Nr1.bmp').convert('L')
    

def extractsubject(subject,PalmOrWrist,Series,LeftOrRight):
    #Labels
    b=np.zeros(50)
    b[int(subject)-1]=1
      
    b=b.reshape(1,50)
    #extracting the pixels
    if LeftOrRight=='Left':
        LoR='L'
    elif LeftOrRight=='Right':
        LoR='R'
    else:
        raise ValueError('error with Left/Right maybe first letter has to be Mayus.')
    if PalmOrWrist=='Palm':
        PoW='P'
    elif LeftOrRight=='Wrist':
        PoW='W'
    else:
        raise ValueError('error with Left/Right maybe first letter has to be Mayus.')


    directory='D:/MSc Artificial Intelligence & Robotics/Biometrics/Project/Database/' + PalmOrWrist + '/o_' + str(subject) + '/' + LeftOrRight + '/Series_' + str(Series) +'/' + PoW + '_o' + str(subject) + '_' + LoR + '_S' + str(Series) + '_Nr1.bmp'
    
    a=extract(directory)
    c=b
    for i in range(2,5):
        directory= directory[:109] + str(i) + '.bmp'
        x=extract(directory)
        a=np.concatenate((a,x))
        c=np.concatenate((c,b))
    return a,c
def binary(identity,testY,trainY):
    BtestY=np.zeros([len(testY),2])
    BtrainY=np.zeros([len(trainY),2])
    for i in range(0,len(testY)):
        if testY[i,identity]==1:
            BtestY[i,0]=1
        else:
            BtestY[i,1]=1
    for ii in range(0,len(trainY)):
        if trainY[ii,identity]==1:
            BtrainY[ii,0]=1
        else:
            BtrainY[ii,1]=1
    return BtestY, BtrainY
def extractdatabase(PalmOrWrist,LeftOrRight,numberofseries):
    trainx=np.array([], dtype=np.int64).reshape(0,3072)
    trainy=np.array([], dtype=np.int64).reshape(0,50)
    testx=np.array([], dtype=np.int64).reshape(0,3072)
    testy=np.array([], dtype=np.int64).reshape(0,50)
    for Series in range (1,numberofseries+1):
        for i in range(1,51):
            subject= str(i).zfill(3)
            a,b=extractsubject(subject,PalmOrWrist,Series,LeftOrRight)
            trainx=np.vstack([trainx,a])
            trainy=np.vstack([trainy,b])
    trainingseries=3-numberofseries
    for Series in reversed(range (4-trainingseries,4)):
        for i in range(1,51):
            subject= str(i).zfill(3)
            a,b=extractsubject(subject,PalmOrWrist,Series,LeftOrRight)
            testx=np.vstack([testx,a])
            testy=np.vstack([testy,b]) 

    ordertrainx=np.array([], dtype=np.int64).reshape(0,3072)
    ordertrainy=np.array([], dtype=np.int64).reshape(0,50)
    ordertestx=np.array([], dtype=np.int64).reshape(0,3072)
    ordertesty=np.array([], dtype=np.int64).reshape(0,50)
    for iii in range(0,4):
        for ii in range(0,100):
#            print(ii*4+iii)
            ordertrainx=np.vstack([ordertrainx,trainx[ii*4+iii]])
            ordertrainy=np.vstack([ordertrainy,trainy[ii*4+iii]])
    for iii1 in range(0,4):
        for ii1 in range(0,50):
            ordertestx=np.vstack([ordertestx,testx[ii1*4+iii1]])
            ordertesty=np.vstack([ordertesty,testy[ii1*4+iii1]])

    return ordertrainx,ordertrainy,ordertestx,ordertesty
aaaa=extract('D:/MSc Artificial Intelligence & Robotics/Biometrics/Project/Database/Palm/o_001/Left/Series_1/P_o001_L_S1_Nr1.bmp')
#a,b=extractsubject('002','Palm','2','Left')
#x,y,xx,yy=extractdatabase('Palm','Left',2)

