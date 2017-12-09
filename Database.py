"""
Artificial Intelligence & Robotics
Biometric systems
Palmvein recognition / Deep Neural Networks
Jose Vicente Jaramillo
Extract library
"""

from PIL import Image, ImageFilter, ImageOps
import cv3
def extract(directory):
    
    #Any feature extraction technique should be implemented here.
    
    #Will read an rgb file, grayscale it, and resize it to 128x96pix 
    #and return a 1D list of gray levels
    #Images are RGB, convert('L') will transform them to grayscale
    #images are too big, resize has been applyed 128x96
    
#    aa=Image.open(directory).resize((128,96)).convert('L')
#    ab= list(aa.getdata())
#    return ab
    
    #edgedettection?
    aa=Image.open(directory).convert('L')
#    aa=equalize(aa)
#    aa.show()
    aa=aa.filter(ImageFilter.BLUR)
#    aa.show()
    aa=ImageOps.autocontrast(aa)
#    aa.show()
    aa=aa.resize((17,12))
#    aa.show()
    aa=aa.filter(ImageFilter.EDGE_ENHANCE_MORE)
    aa=aa.filter(ImageFilter.FIND_EDGES)
    ab= list(aa.getdata())
#    aa.show()
    return ab

#extract('Database/Palm/o_001/Left/Series_1/P_o001_L_S1_Nr1.bmp')

    

def extractsubject(subject,PalmOrWrist,Series,LeftOrRight):
    #Labels
    y=[0]*50
    y[int(subject)-1]=1
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
    a=[]
    b=[]

    directory='Database/' + PalmOrWrist + '/o_' + str(subject) + '/' + LeftOrRight + '/Series_' + str(Series) +'/' + PoW + '_o' + str(subject) + '_' + LoR + '_S' + str(Series) + '_Nr1.bmp'
    for i in range(1,5):
        directory= directory[:48] + str(i) + '.bmp'
        x=extract(directory)
        a.append(x)
        b.append(y)
    return a,b

def extractdatabase(PalmOrWrist,LeftOrRight,numberofseries):
    x=[]
    y=[]
    xx=[]
    yy=[]
    for Series in range (1,numberofseries+1):
        for i in range(1,51):
            subject= str(i).zfill(3)
            a,b=extractsubject(subject,PalmOrWrist,Series,LeftOrRight)
            x.extend(a)
            y.extend(b)
    trainingseries=3-numberofseries
    for Series in reversed(range (4-trainingseries,4)):
        for i in range(1,51):
            subject= str(i).zfill(3)
            a,b=extractsubject(subject,PalmOrWrist,Series,LeftOrRight)
            xx.extend(a)
            yy.extend(b)
    return x,y,xx,yy

