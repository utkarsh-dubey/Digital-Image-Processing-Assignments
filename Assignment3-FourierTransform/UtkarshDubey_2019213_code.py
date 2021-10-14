import numpy as np
from math import *
import cv2
import matplotlib.pyplot as plt
from random import *
import scipy.signal as ss


def getValue(D0,i,j,m,n):
    num=((i-m)**2+(j-n)**2)**(0.5)
    ans=num/D0
    ans=(1+ans**4)
    ans=1/ans
    return ans

def butterworth():
    img = cv2.imread("input.jpg", 0)
    img = cv2.resize(img,(256,256))
    cv2.imshow("Input Image", img)
    cv2.waitKey()
    inputImage = np.array(img)
    m,n=inputImage.shape
    imagePad=np.zeros((2*m,2*n))
    for i in range(m):
        for j in range(n):
            imagePad[i][j]=inputImage[i][j]*((-1)**(i+j))
    imageDft=np.fft.fft2(imagePad)
    D0 = np.array([10, 30, 60])

    h=np.zeros((2*m,2*n))
    for d in D0:
        for i in range(2*m):
            for j in range(2*n):
                h[i][j]=getValue(d,i,j,m,n)

        outputImage=np.multiply(h,imageDft)
        outputImage1=np.fft.ifft2(outputImage)
        outputImage1=outputImage1.real
        check=outputImage1
        for i in range(2*m):
            for j in range(2*n):
                outputImage1[i][j]=outputImage1[i][j]*((-1)**(i+j))
        finalOutputImage=np.abs(outputImage1)
        finalOutputImage=finalOutputImage[:m,:n]
        finalOutputImage=np.array(finalOutputImage,'uint8')
        cv2.imshow("Blurred Image D0 = "+str(d),finalOutputImage)
    cv2.waitKey()
    cv2.destroyAllWindows()



def dft():

    img = cv2.imread("input.jpg", 0)
    cv2.imshow("Input Image", img)
    cv2.waitKey()
    inputImage = np.array(img)

    boxFilter=(1/81)*np.ones((9,9))
    image1=np.zeros((inputImage.shape[0]+8,inputImage.shape[1]+8))
    image2=np.zeros((inputImage.shape[0]+8,inputImage.shape[1]+8))
    for i in range(inputImage.shape[0]):
        for j in range(inputImage.shape[1]):
            if(i<9 and j<9):
                image1[i][j]=boxFilter[i][j]
            # if(i<inputImage.shape[0] and j<inputImage.shape[1]):
            image2[i][j]=inputImage[i][j]

    fourier1=np.fft.fft2(image1)
    fourier2=np.fft.fft2(image2)

    outputImage=np.ones((inputImage.shape[0]+8,inputImage.shape[1]+8),dtype = 'complex_')

    # for i in range(inputImage.shape[0]+8):
    #     for j in range(inputImage.shape[1]+8):
    #         outputImage[i][j]=np.multiply(fourier1[i][j],fourier2[i][j])
    outputImage = np.multiply(fourier1, fourier2)
    outputImage1=np.fft.ifft2(outputImage)
    outputImage1=outputImage1.real
    finalOutputImage=np.array(outputImage1,np.uint8)

    # for i in range(inputImage.shape[0]):
    #     for j in range(inputImage.shape[1]):
    #         finalOutputImage[i][j]=outputImage1[i][j]

    cv2.imshow("Output Image", finalOutputImage)
    cv2.waitKey()

    spaImg = ss.convolve2d(inputImage, boxFilter)
    spaImg = np.array(spaImg, dtype='uint8')
    # inbuiltOutput=cv2.filter2D(src=inputImage,ddepth=-1,kernel=boxFilter)
    cv2.imshow("inbuilt output", spaImg)
    cv2.waitKey()



def denoise():
    img = cv2.imread("noiseIm.jpg", 0)
    cv2.imshow("Input Image", img)
    # cv2.waitKey()

    inputImage = np.array(img)
    m,n=inputImage.shape
    for i in range(m):
        for j in range(n):
            inputImage[i][j]=inputImage[i][j]*((-1)**(i+j))
    dft=np.fft.fft2(inputImage)
    # centredDft=np.fft.fftshift(dft)
    # print(centredDft)
    # spectrum=np.log(centredDft)
    # print(spectrum)
    boxFilter=np.ones(inputImage.shape)

    for i in range(m):
        for j in range(n):
            if(i>95 and i<99 and j>95 and j<99):
                boxFilter[i][j]=0
                boxFilter[m-i][n-j]=0

    output=np.multiply(dft,boxFilter)
    idft=np.fft.ifft2(output)
    idft=idft.real

    for i in range( m):
        for j in range( n):
            idft[i][j] = idft[i][j] * ((-1) ** (i + j))

    spectrumInput=np.array(np.log(1+np.abs(np.double(dft))).real,np.uint8)
    finalOutput=np.array(idft,np.uint8)
    spectrum=np.log(1+np.abs(np.fft.fft2(idft))).real
    cv2.imshow("input magnitude spectrum", spectrumInput)
    cv2.imshow("Output Image", finalOutput)
    cv2.imshow("output magnitude spectrum", spectrum)
    cv2.imwrite("centered.jpg", spectrum)
    cv2.waitKey()



#
# butterworth()   #question 1
# # dft()       #question 3
denoise()   #question 4