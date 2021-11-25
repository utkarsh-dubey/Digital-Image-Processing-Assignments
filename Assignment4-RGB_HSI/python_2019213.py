import numpy as np
from math import *
import cv2
import matplotlib.pyplot as plt
from random import *
import scipy.signal as ss

def histogramEqualisation(inputImage1):

    # img=cv2.imread("input.jpg", 0)
    # cv2.imshow("Input Image", img)
    # cv2.waitKey()
    #
    # inputImage=np.array(img)
    inputImage=np.array(inputImage1,dtype='uint8')
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
        s[i]=floor(H[i]*255)

    outImage=np.zeros(inputImage.shape)

    for i in range(inputImage.shape[0]):
        for j in range(inputImage.shape[1]):
            outImage[i][j]=s[inputImage[i][j]]

    return outImage
    # cv2.imshow("Output Image",outImage)
    # cv2.waitKey()
    #
    # outHisto=np.zeros(256)
    #
    # for i in range(256):
    #     freq=np.where(outImage==i)
    #     outHisto[i]=len(freq[0])/(outImage.shape[0]*outImage.shape[1])

    # plt.figure("Input")
    # plt.subplot(121)
    # plt.plot(h,'.')
    # plt.title("Histogram for input image")
    # plt.xlabel("Pixel value (r)")
    # plt.ylabel("P(r)")
    #
    # plt.subplot(122)
    # plt.imshow(inputImage)
    # plt.title("Input image")
    #
    # plt.show()
    #
    # plt.figure("Output")
    # plt.subplot(121)
    # plt.plot(outHisto, '.')
    # plt.title("Histogram for output image")
    # plt.xlabel("Pixel value (s)")
    # plt.ylabel("P(s)")
    #
    # plt.subplot(122)
    # plt.imshow(outImage)
    # plt.title("Output image")
    #
    # plt.show()


def question1():
    img = cv2.imread("noiseIm.jpg", 0)
    cv2.imshow("Input Image", img)
    cv2.waitKey()
    inputImage = np.array(img)
    m,n=inputImage.shape
    boxFilter=np.zeros((m+11,n+11));
    for i in range(11):
        for j in range(11):
            boxFilter[i][j]=1/(11**2)
    print(boxFilter)
    # print(inputImage.shape)
    paddedImage=np.zeros((m+11,n+11),dtype=float)
    for i in range(m):
        for j in range(n):
            paddedImage[i][j]=inputImage[i][j]

    laplacianMask=[
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]
    ]

    paddedLaplaplacianMask=np.zeros((m+11,n+11), dtype=float)

    for i in range(len(laplacianMask)):
        for j in range(len(laplacianMask[0])):
            paddedLaplaplacianMask[i][j]=laplacianMask[i][j]

    imageFourier=np.fft.fft2(paddedImage)
    laplacianMaskFourier=np.fft.fft2(paddedLaplaplacianMask)
    boxFilterFourier=np.fft.fft2(boxFilter)

    boxFilterFourierConj=np.conj(boxFilterFourier)
    boxFilterFourierMagnitude=np.abs(boxFilterFourier)
    laplacianMaskFourierMagnitude=np.abs(laplacianMaskFourier)

    calculations=boxFilterFourierConj

    calculations/=(boxFilterFourierMagnitude**2+((laplacianMaskFourierMagnitude**2)/2))
    calculations *= imageFourier
    output=np.fft.ifft2(calculations)
    output=output.real

    # finalOutput=np.zeros((m,n), dtype='uint8')
    # for i in range(m):
    #     for j in range(n):
    #         finalOutput[i][j]=output[i][j]
    finalOutput=output[0:m,0:n]
    imgOri = cv2.imread("input.jpg", 0)
    # cv2.imshow("Input Image", img)
    # cv2.waitKey()
    imageOriginal = np.array(imgOri)
    MSE=np.mean((imageOriginal-finalOutput)**2)
    psnr=10*log10((255**2)/MSE)

    print("PSNR for the best restored image = ",psnr)
    plt.title("Restored image")
    plt.imshow(finalOutput,"gray")
    # cv2.waitKey()
    plt.show()

def question3():
    img = cv2.imread("dipass4.tif")
    cv2.imshow("Input Image", img)
    # cv2.waitKey()
    inputImage = np.array(img,dtype=float)/255
    # print(inputImage)
    # b, g, r = cv2.split(img)
    r=inputImage[:,:,0]
    g=inputImage[:,:,1]
    b=inputImage[:,:,2]

    # print(b)
    # print(g)

    num=((r-g)+(r-b))/2
    deno=((r-g)**2+((r-b)*(g-b)))**(1/2)
    print(b.shape)
    H=(np.arccos(num/(deno+1e-7)))*(180/pi)
    # H[b>g]=360-H[b-g]
    # print(H)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if(b[i][j]>g[i][j]):
                H[i][j]=360-H[i][j]

    H=H/360
    # print(H)
    # S=1-(3/)
    # S=1-3*(np.min(r,g,b))/(r+g+b)
    S=np.zeros(b.shape)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            S[i][j]=1-3*min(r[i][j],g[i][j],b[i][j])/(r[i][j]+g[i][j]+b[i][j]+1e-7)

    # print(S)

    I=(r+g+b)/3
    # print(I)
    HSI=np.zeros(inputImage.shape)
    HSI[:,:,0]=H
    HSI[:,:,1]=S
    HSI[:,:,2]=I
    # print(HSI)
    cv2.imshow("HSI",HSI)
    I*=255
    I2=histogramEqualisation(I)/255
    # print(I2)
    HSI2 = np.zeros(inputImage.shape)
    HSI2[:, :, 0] = H
    HSI2[:, :, 1] = S
    HSI2[:, :, 2] = I2
    cv2.imshow("HSI2", HSI2)
    r2=np.zeros(r.shape)
    b2=np.zeros(b.shape)
    g2=np.zeros(g.shape)
    H*=360
    # print("hhhhhhh",H)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if(H[i][j]>=0 and H[i][j]<120):
                b2[i][j]=I2[i][j]*(1-S[i][j])
                r2[i][j]=I2[i][j]*(1+(S[i][j]*np.cos(radians(H[i][j]))*180/pi)/(np.cos(pi/3-radians(H[i][j]))*180/pi))
                g2[i][j]=3*I2[i][j]-(r2[i][j]+b2[i][j])
            elif(H[i][j]>=120 and H[i][j]<240):
                H[i][j]=H[i][j]-120
                r2[i][j]=I2[i][j]*(1-S[i][j])
                g2[i][j]=I2[i][j]*(1+(S[i][j]*np.cos(radians(H[i][j]))*180/pi)/(np.cos(pi/3-radians(H[i][j]))*180/pi))
                b2[i][j] = 3 * I2[i][j] - (r2[i][j] + g2[i][j])
            elif(H[i][j]>=240 and H[i][j]<=360):
                H[i][j]=H[i][j]-240
                g2[i][j] = I2[i][j] * (1 - S[i][j])
                b2[i][j] = I2[i][j] * (1 + (S[i][j] *np.cos(radians(H[i][j]))*180/pi)/(np.cos(pi/3-radians(H[i][j]))*180/pi))
                r2[i][j] = 3 * I2[i][j] - (g2[i][j] + b2[i][j])

    RBG=np.zeros(HSI2.shape)
    print(HSI2.shape)
    RBG[:, :, 0] = r2
    RBG[:, :, 1] = g2
    RBG[:, :, 2] = b2
    # print(RBG)
    # RBG=np.array(RBG,np.uint8)
    cv2.imshow("Final image",RBG)


    cv2.waitKey()





# question1()     #question 1
question3()     #question 3