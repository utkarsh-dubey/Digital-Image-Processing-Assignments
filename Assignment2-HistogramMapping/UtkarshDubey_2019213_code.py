import numpy as np
from math import *
import cv2
import matplotlib.pyplot as plt
from random import *


def histogramEqualisation():

    img=cv2.imread("input.jpg", 0)
    cv2.imshow("Input Image", img)
    cv2.waitKey()

    inputImage=np.array(img)
    inputSize=inputImage.shape[0]*inputImage.shape[1]

    h=np.zeros(256)

    for i in range(256):
        freq=np.where(inputImage==i)
        h[i]=len(freq[0])/inputSize

    H=np.zeros(256)
    H[0]=h[0]

    for i in range(1,256):
        H[i]=h[i]+H[i-1]

    s=np.zeros(256)

    for i in range(256):
        s[i]=H[i]*255

    outImage=np.zeros(inputImage.shape,np.uint8)

    for i in range(inputImage.shape[0]):
        for j in range(inputImage.shape[1]):
            outImage[i][j]=s[inputImage[i][j]]

    cv2.imshow("Output Image",outImage)
    cv2.waitKey()

    outHisto=np.zeros(256)

    for i in range(256):
        freq=np.where(outImage==i)
        outHisto[i]=len(freq[0])/(outImage.shape[0]*outImage.shape[1])

    plt.figure("Input")
    plt.subplot(121)
    plt.plot(h,'.')
    plt.title("Histogram for input image")
    plt.xlabel("Pixel value (r)")
    plt.ylabel("P(r)")

    plt.subplot(122)
    plt.imshow(inputImage)
    plt.title("Input image")

    plt.show()

    plt.figure("Output")
    plt.subplot(121)
    plt.plot(outHisto, '.')
    plt.title("Histogram for output image")
    plt.xlabel("Pixel value (s)")
    plt.ylabel("P(s)")

    plt.subplot(122)
    plt.imshow(outImage)
    plt.title("Output image")

    plt.show()




def histogramMatching():

    # input image
    # image = input("Enter the name of the image with extension (make sure it is present in the root folder)\n")
    img = cv2.imread("input.jpg", 0)  # reading the image as greysfactorale

    # preview of the input image
    cv2.imshow("Input Image", img)
    cv2.waitKey()

    targetImage=np.array(255*(img/255)**0.5,dtype='uint8')

    cv2.imshow("Target Image", targetImage)
    cv2.waitKey()
    img=np.array(img)
    hashy={}
    hashy2={}
    sizeInput=img.shape[0]*img.shape[1]
    sizeTarget=targetImage.shape[0]*targetImage.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] not in hashy):
                hashy[img[i][j]]=1
            else:
                hashy[img[i][j]]+=1
    for i in range(targetImage.shape[0]):
        for j in range(targetImage.shape[1]):
            if(targetImage[i][j] not in hashy2):
                hashy2[targetImage[i][j]]=1
            else:
                hashy2[targetImage[i][j]]+=1
    # print(hashy)
    h=np.zeros((256,1))
    g=np.zeros((256,1))
    for i in range(256):
        freq=np.where(img==i)
        h[i]=len(freq[0])/sizeInput

    for i in range(256):
        freq=np.where(targetImage==i)
        g[i]=len(freq[0])/sizeInput

    H=np.zeros(256)
    H[0]=h[0]

    for i in range(1,256):
        H[i]=h[i]+H[i-1]

    G=np.zeros(256)
    G[0]=g[0]

    for i in range(1,256):
        G[i]=g[i]+G[i-1]


    mapping=np.zeros(256)

    for i in range(256):
        mapping[i]=np.abs(H[i]-G).argmin()
    # print(mapping)
    outImage=np.zeros(img.shape,np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            outImage[i][j]=mapping[img[i][j]]

    cv2.imshow("Output image",outImage)
    cv2.waitKey()

    outHisto = np.zeros(256)

    for i in range(256):
        freq = np.where(outImage == i)
        outHisto[i] = len(freq[0]) / (outImage.shape[0] * outImage.shape[1])

    plt.figure("Input")
    plt.subplot(121)
    plt.plot(h, '.')
    plt.title("Histogram for input image")
    plt.xlabel("Pixel value (r)")
    plt.ylabel("P(r)")

    plt.subplot(122)
    plt.imshow(img)
    plt.title("Input image")

    plt.show()

    plt.figure("Target")
    plt.subplot(121)
    plt.plot(g, '.')
    plt.title("Histogram for target image")
    plt.xlabel("Pixel value (t)")
    plt.ylabel("P(t)")

    plt.subplot(122)
    plt.imshow(targetImage)
    plt.title("Target image")

    plt.show()

    plt.figure("Output")
    plt.subplot(121)
    plt.plot(outHisto, '.')
    plt.title("Histogram for output image")
    plt.xlabel("Pixel value (s)")
    plt.ylabel("P(s)")

    plt.subplot(122)
    plt.imshow(outImage)
    plt.title("Output image")

    plt.show()


def printMatrix(a):
    for i in range(len(a)):
        for j in range(len(a[0])):
            print(a[i][j],end=" ")
        print()


def convolution():

    #for random input
    # image = [[randint(1,20)]*3 for i in range(3)]
    # filter = [[randint(1,20)]*3 for i in range(3)]

    #for user input
    image=[]
    for i in range(3):
        image.append(list(map(int,input().split())))
    filter=[]
    for i in range(3):
        filter.append(list(map(int,input().split())))

    #rotating the filter matrix
    filterRotated=[[0]*3 for i in range(3)]
    for i in range(3):
        for j in range(3):
            filterRotated[i][j]=filter[2-i][2-j]

    midOutput=[[0]*5 for i in range(5)]
    for i in range(1,4):
        for j in range(1,4):
            midOutput[i][j]=image[i-1][j-1]

    output=[[0]*5 for i in range(5)]

    for i in range(5):
        for j in range(5):
            for k in range(3):
                for l in range(3):
                    if((i+k-1>=0 and i+k-1<5 and j+l-1>=0 and j+l-1<5)):
                        output[i][j]+=midOutput[i+k-1][j+l-1]*filterRotated[k][l]


    print("Input Matrix")
    printMatrix(image)
    print("Filter Matrix")
    printMatrix(filter)
    print("Rotated Filter Matrix")
    printMatrix(filterRotated)
    print("Output Matrix")
    printMatrix(output)



# histogramEqualisation()  #question 3
histogramMatching()   #question 4
# convolution()      #question 5