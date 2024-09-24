# -*- coding: utf-8 -*-
"""
modified from https://github.com/ritikajha/ECG-Digitization/tree/main

"""

#  organizing imports
import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from PIL import Image
# path to input image is specified and
# image is loaded with imread command
import fitz  # PyMuPDF

# Open the PDF file

pdf_document = "ecg-20240730-101924.pdf"
if len(sys.argv) > 1:
   pdf_document = sys.argv[1]
   
doc = fitz.open(pdf_document)

# Select page
page = doc.load_page(1)  # Page 2 (zero-based index)

# Define the area you want to capture - (x1, y1, x2, y2)
rect = fitz.Rect(23, 100, 588, 700)  # example coordinates

# Render page to a pixmap
zoom_x = 2.0  # Increase the zoom factor for higher resolution
zoom_y = 2.0
matrix = fitz.Matrix(zoom_x, zoom_y)
pix = page.get_pixmap(matrix=matrix,clip=rect)
# Save as an image
output = "output.png"
pix.pil_save(output)

image1 = cv2.imread('output.png') #TODO: use pil from bytes

# cv2.cvtColor is applied over the
# image input with applied parameters
# to convert the image in grayscale
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# applying Otsu thresholding
# as an extra flag in binary
# thresholding
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY +
                                            cv2.THRESH_OTSU)

# the window showing output image
# with the corresponding thresholding
# techniques applied to the input image


threshold_values = {}
h = [1]


def Hist(img):
   row, col = img.shape
   y = np.zeros(256)
   for i in range(0,row):
      for j in range(0,col):
         y[img[i,j]] += 1
   x = np.arange(0,256)
   #plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
   #plt.show()
   return y


def regenerate_img(img, threshold):
    row, col = img.shape
    y = np.zeros((row, col))
    for i in range(0,row):
        for j in range(0,col):
            if img[i,j] >= threshold:
                y[i,j] = 255
            else:
                y[i,j] = 0
    return y



def countPixel(h):
    cnt = 0
    for i in range(0, len(h)):
        if h[i]>0:
           cnt += h[i]
    return cnt


def wieght(s, e):
    w = 0
    for i in range(s, e):
        w += h[i]
    return w


def mean(s, e):
    m = 0
    w = wieght(s, e)
    for i in range(s, e):
        m += h[i] * i

    return m/float(w)


def variance(s, e):
    v = 0
    m = mean(s, e)
    w = wieght(s, e)
    for i in range(s, e):
        v += ((i - m) **2) * h[i]
    v /= w
    return v


def threshold(h):
    cnt = countPixel(h)
    for i in range(1, len(h)):
        vb = variance(0, i)
        wb = wieght(0, i) / float(cnt)
        mb = mean(0, i)

        vf = variance(i, len(h))
        wf = wieght(i, len(h)) / float(cnt)
        mf = mean(i, len(h))

        V2w = wb * (vb)*(vb) + wf * (vf)*(vf)
        V2b = wb * wf * (mb - mf)**2

        fw = open("trace.txt", "a")
        fw.write('T='+ str(i) + "\n")

        fw.write('Wb='+ str(wb) + "\n")
        fw.write('Mb='+ str(mb) + "\n")
        fw.write('Vb='+ str(vb) + "\n")

        fw.write('Wf='+ str(wf) + "\n")
        fw.write('Mf='+ str(mf) + "\n")
        fw.write('Vf='+ str(vf) + "\n")

        fw.write('within class variance='+ str(V2w) + "\n")
        fw.write('between class variance=' + str(V2b) + "\n")
        fw.write("\n")

        if not math.isnan(V2w):
            threshold_values[i] = V2w


def get_optimal_threshold():
    min_V2w = min(threshold_values.values())
    optimal_threshold = [k for k, v in threshold_values.items() if v == min_V2w]
    #print ('optimal threshold', optimal_threshold[0])
    return optimal_threshold[0]


image = Image.open('output.png').convert("L")
img = np.asarray(image)

h = Hist(img)
threshold(h)
op_thres = get_optimal_threshold()

res = regenerate_img(img, op_thres)

#plt.figure(figsize=(10, 10))
#plt.imshow(res,cmap="gray")
#plt.savefig("otsu.jpg")

#img = cv.imread('e.png')
img=res
img = np.full((1130,1566), 6, np.uint8)
converted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
dst = cv.fastNlMeansDenoisingColored(converted_img,None,10,10,7,21)

#cap = cv.VideoCapture('vtest.avi')
# create a list of first 5 frames
#img = [cap.read()[1] for i in xrange(5)]
# convert all to grayscale
i=res
#gray = cv.cvtColor(i, cv.COLOR_BGR2GRAY)
# convert all to float64
gray = np.float64(i)
# create a noise of variance 25
noise = np.random.randn(*gray[1].shape)*10
# Add this noise to images
noisy = i+noise
# Convert back to uint8
noisy = np.uint8(np.clip(i,0,255))
# Denoise 3rd frame considering all the 5 frames
#dst = cv.fastNlMeansDenoisingMulti(noisy, 2, 5, 1, 4, 7, 5)
dst=cv.fastNlMeansDenoising(noisy,dst,3,7,21)
# plt.subplot(131),plt.imshow(gray,'gray')
# plt.subplot(132),plt.imshow(noisy,'gray')
# plt.subplot(133),plt.imshow(dst,'gray')
# plt.show()

#plt.imshow(gray,'gray')

#plt.imshow(noisy,'gray')

#plt.imshow(dst,'gray')

#image = cv2.imread("test.jpg")
image=dst
crop_img = image[25:1180, 0:1130]
#plt.imshow(crop_img,cmap="gray")

"""This removes the grid dots, somehow"""
img=crop_img
for i in range (0,crop_img.shape[0]-10):
  for j in range (0,crop_img.shape[1]-10):
    if img[i][j]==0:
      #check
      count=0
      for k in range (i-5,i+5):
        for l in range (j-5,j+5):
          if img[k][l]==0:
            count=count+1
      if(count<5):
        img[i][j]=255

#plt.imshow(img,cmap="gray")
#cv2.imwrite('cleaned.png',img)

"""###DIGITIZING"""


I = img#cv2.imread("cleaned.jpg",0)

pixel_from_bottom=[]
#plt.imshow(I,cmap="gray")
for sprint in range(4):
   match sprint:
      case 0:
         crop_img = I[0:125, 100:1130]
      case 1:
         crop_img = I[340:340+125, 0:1130]
      case 2:
         crop_img = I[680:680+125, 0:1130]
      case 3:
         crop_img = I[1011:1011+125, 0:925]
   #plt.imshow(crop_img,cmap="gray")

   #cv2.imwrite('crop.jpg',crop_img)
   img = crop_img
   '''
   match sprint:
      case 0:
         FUHaxes = [91,92] #TODO: row 92 has straglers, do special; killStagler that deals with dots along H axis vestige
      case 1:
         FUHaxes = [92,91] #91 for stragglers
      case 2:
         FUHaxes = [91,92] #91 for stragglers
      case 3:
         FUHaxes = [101,100]
   '''
   FUHaxes = np.array([91,92]) + (sprint==3)*np.array([10,8])
   for i in range(len(img[0])):
     for y in FUHaxes:
        img[y][i] = 255
   #print(f'sprint {sprint}')
   match sprint:
      case 0:
         FUVaxes = [40] #TODO: row 92 has straglers, do special; killStagler that deals with dots along H axis vestige
      case 1:
         FUVaxes = [92,91] #91 for stragglers
      case 2:
         FUVaxes = [91,92] #91 for stragglers
      case 3:
         FUVaxes = [140,282, 424, 565, 707, 849]
   for row in FUVaxes:
       img[:, row] = 255
   
   if sprint == 4:
      plt.imshow(img,cmap="gray")
      plt.show()
      exit()
   #plt.imshow(img,cmap="gray")

   pixel_from_top=[]
   for i in range (0,len(img[0])):
     id=0

     for j in range (0,124):
         if img[j][i]==0:# and img[j+10][i] == 255:
           if j != 0: pixel_from_top.append(j)
           # print("f\n")
           break

   
   for i in range(0,len(pixel_from_top)):
     pixel_from_bottom.append(120-pixel_from_top[i] + 8*(sprint==3))

   # for i in range (0,301,100):
   #   plt.axvline(x=i)

   
for ind,num in enumerate(pixel_from_bottom):
  if num > 100:
    pass#pixel_from_bottom[ind] = pixel_from_bottom[ind-1]

plt.figure(figsize=(25, 5))
plt.xticks(range(0,len(pixel_from_bottom),50))
plt.plot(pixel_from_bottom)
plt.show()
with open(pdf_document.replace('.pdf','.txt'),'a') as txt:
   for num in pixel_from_bottom:
      txt.write(str(num)+'\n')
   txt.close
