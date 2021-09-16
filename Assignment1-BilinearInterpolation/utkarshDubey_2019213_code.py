import cv2
import numpy as np
from math import *

def bilinear():

    #input image
    image=input("Enter the name of the image with extension (make sure it is present in the root folder)\n")
    img=cv2.imread(image,0)   #reading the image as greysfactorale

    #preview of the input image
    cv2.imshow("Preview",img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    #input interpolation factor
    print("Enter interpolation factor")
    factor=int(input())

    imageArr=np.array(img)
    size=imageArr.shape
    m=size[0]   #input image size
    n=size[1]   #input image size
    M=(m-1)*factor
    N=(n-1)*factor
    # print(n,m)
    imgOut=np.zeros((int(m*factor),int(n*factor)),np.uint8)
    # print(imgOut)
    for i in range(M+1):
        for j in range(N+1):
            if(imgOut[i][j]==0):
                x=i/factor
                y=j/factor
                if(ceil(x)!=x):
                    x1=floor(x)
                    x2=ceil(x)
                else:
                    if(x==0):
                        x1=0
                        x2=1
                    else:
                        x1=x-1
                        x2=x

                if (ceil(y) != y):
                    y1 = floor(y)
                    y2 = ceil(y)
                else:
                    if (y == 0):
                        y1 = 0
                        y2 = 1
                    else:
                        y1 = y - 1
                        y2 = y
                
                x1=int(x1)
                x2=int(x2)
                y1=int(y1)
                y2=int(y2)
                
                X=[[x1,y1,x1*y1,1],[x2,y1,x2*y1,1],[x1,y2,x1*y2,1],[x2,y2,x2*y2,1]]
                Y=[[imageArr[x1][y1]],[imageArr[x2][y1]],[imageArr[x1][y2]],[imageArr[x2][y2]]]
                # print("check",X)
                Xinv=np.linalg.inv(X)
                A=np.dot(Xinv,Y)
                imgOut[i][j]=np.dot(np.array([x,y,x*y,1]),A)


    #mirroring for padding

    for i in range(M+1):
        for j in range(N+1,len(imgOut[0])):
            imgOut[i][j]=max(0,imgOut[i][j-1])

    for i in range(M+1,len(imgOut)):
        for j in range(len(imgOut[0])):
            imgOut[i][j]=max(0,imgOut[i-1][j])

    cv2.imshow("Output Image",imgOut)
    cv2.waitKey()
    cv2.destroyAllWindows()


def geometric():
    # input image
    image = input("Enter the name of the image with extension (make sure it is present in the root folder)\n")
    img = cv2.imread(image, 0)  # reading the image as greysfactorale

    inputImp=np.array(img)
    size=inputImp.shape
    M=size[0]
    N=size[1]

    imageOut=np.ones((300,300),np.uint8)

    C1=[[1,0,0],[0,1,0],[60,70,1]]

    for i in range(M):
        for j in range(N):
            temp=np.dot([i,j,1],C1)
            imageOut[int(temp[0])][int(temp[1])]=inputImp[i][j]

    cv2.imshow('input',imageOut)
    cv2.waitKey()
    cv2.imwrite("reference.jpg",imageOut)

    imageOut=np.ones((300,300),np.uint8)

    rotate=np.zeros((3,3),float)
    print("Enter the degree of rotation")
    angle=float(input())
    rotate[0][0]=cos((angle/180)*pi)
    rotate[1][0]=sin((angle/180)*pi)
    rotate[0][1]=sin((angle/180)*pi)*(-1)
    rotate[1][1]=cos((angle/180)*pi)
    rotate[2][2]=1

    print("Translate x by :")
    translateX = float(input())
    print("Translate y by :")
    translateY = float(input())
    trans = [[1, 0, 0], [0, 1, 0], [translateX, translateY, 1]]

    print("Scale x by :")
    scaleX=float(input())
    print("Scale y by :")
    scaleY=float(input())

    scale=[[scaleX,0,0],[0,scaleY,0],[0,0,1]]

    T=np.dot(np.dot(rotate,scale),trans)
    invT=np.linalg.inv(T)

    print("Transformation Matrix :")
    print(T)

    minX=0
    maxX=0
    minY=0
    maxY=0

    for i in range(M):
        for j in range(N):
            temp=np.dot([i,j,1],T)
            minX=min(minX,temp[0])
            minY=min(minY,temp[1])
            maxX=max(maxX,temp[0])
            maxY=max(maxY,temp[1])

    for i in range(int(minX),int(maxX)):
        for j in range(int(minY),int(maxY)):
            temp=np.dot([i,j,1],invT)
            x=temp[0]
            y=temp[1]

            #overflow conditions
            if(np.logical_or(x<0,x>M-1).all()):
                continue
            elif(np.logical_or(y<0,y>N-1).all()):
                continue

            if (ceil(x) != x):
                x1 = floor(x)
                x2 = ceil(x)
            else:
                if (x == 0):
                    x1 = 0
                    x2 = 1
                else:
                    x1 = x - 1
                    x2 = x

            if (ceil(y) != y):
                y1 = floor(y)
                y2 = ceil(y)
            else:
                if (y == 0):
                    y1 = 0
                    y2 = 1
                else:
                    y1 = y - 1
                    y2 = y

            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            X = [[x1, y1, x1 * y1, 1], [x2, y1, x2 * y1, 1], [x1, y2, x1 * y2, 1], [x2, y2, x2 * y2, 1]]
            Y = [[inputImp[x1][y1]], [inputImp[x2][y1]], [inputImp[x1][y2]], [inputImp[x2][y2]]]

            invX=np.linalg.inv(X)
            A=np.dot(invX,Y)

            temp2=np.dot([i,j,1],C1)

            imageOut[int(temp2[0])][int(temp2[1])]=np.dot(np.array([x,y,x*y,1]), A)

    cv2.imshow("output image",imageOut)
    cv2.imwrite("unregistered.jpg",imageOut)
    cv2.waitKey()
    cv2.destroyAllWindows()

def registered():

    reference=cv2.imread("reference.jpg",0)
    cv2.imshow("reference image",reference)
    cv2.waitKey()
    referenceImage=np.array(reference)

    unregistered=cv2.imread("unregistered.jpg",0)
    cv2.imshow("unregistered image",unregistered)
    cv2.waitKey()
    unregisteredImage=np.array(unregistered)


    x11,y11=115,75
    x12,y12=140,176

    x21,y21=106,68
    x22,y22=138,151

    x31,y31=91,101
    x32,y32=72,178

    X=[[x12,y12,1],[x22,y22,1],[x32,y32,1]]
    V=[[x11,y11,1],[x21,y21,1],[x31,y31,1]]

    T=np.dot(np.linalg.inv(V),X)
    invT=np.linalg.inv(T)

    Mref,Nref=referenceImage.shape
    Munref,Nunreg=unregisteredImage.shape

    registeredImage=np.zeros((Mref,Nref),np.uint8)

    for i in range(Mref):
        for j in range(Nref):
            temp=np.dot([i,j,1],T)
            x=temp[0]
            y=temp[1]

            # overflow conditions
            if (np.logical_or(x < 0, x > Mref - 1).all()):
                continue
            elif (np.logical_or(y < 0, y > Nref - 1).all()):
                continue

            if (ceil(x) != x):
                x1 = floor(x)
                x2 = ceil(x)
            else:
                if (x == 0):
                    x1 = 0
                    x2 = 1
                else:
                    x1 = x - 1
                    x2 = x

            if (ceil(y) != y):
                y1 = floor(y)
                y2 = ceil(y)
            else:
                if (y == 0):
                    y1 = 0
                    y2 = 1
                else:
                    y1 = y - 1
                    y2 = y

            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            X = [[x1, y1, x1 * y1, 1], [x2, y1, x2 * y1, 1], [x1, y2, x1 * y2, 1], [x2, y2, x2 * y2, 1]]
            Y = [[unregisteredImage[x1][y1]], [unregisteredImage[x2][y1]], [unregisteredImage[x1][y2]], [unregisteredImage[x2][y2]]]

            Xinv = np.linalg.inv(X)
            A = np.dot(Xinv, Y)
            registeredImage[i][j] = np.dot(np.array([x, y, x * y, 1]), A)

    cv2.imshow("registered image",registeredImage)
    cv2.waitKey()
    cv2.imwrite("registered.jpg",registeredImage)
    cv2.destroyAllWindows()


# bilinear()  #question 3

# geometric()   #question 4

registered()    #question 5




