# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html

import cv2
import numpy as np
from matplotlib import pyplot as plt

path='../ipcam_images/train/empty/'

#empty night
img_name = 'ipcam_2018-06-13_22.19.53.923.jpg'
img = cv2.imread(path + img_name,1)
#cv2.imshow(path + 'ipcam_2018-06-14_11.58.45.536.jpg',img)
#plt.hist(img.ravel())
#plt.show()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/empty_night-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-13_23.52.09.549.jpg'
img = cv2.imread(path + img_name,1)
#cv2.imshow(path + 'ipcam_2018-06-14_11.58.45.536.jpg',img)
#plt.hist(img.ravel())
#plt.show()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/empty_night-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-14_04.01.08.495.jpg'
img = cv2.imread(path + img_name,1)
#cv2.imshow(path + 'ipcam_2018-06-14_11.58.45.536.jpg',img)
#plt.hist(img.ravel())
#plt.show()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/empty_night-' + img_name, dpi=150)
plt.close()

#empty daylight morning
img_name = 'ipcam_2018-06-15_07.45.27.029.jpg'
img = cv2.imread(path + img_name,1)
#cv2.imshow(path + 'ipcam_2018-06-14_11.58.45.536.jpg',img)
#plt.hist(img.ravel(),256,[0,256], log=bool)
#plt.hist(img.ravel())
#plt.show()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/empty_daylight-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-16_09.23.49.682.jpg'
img = cv2.imread(path + img_name,1)
#cv2.imshow(path + 'ipcam_2018-06-14_11.58.45.536.jpg',img)
#plt.hist(img.ravel(),256,[0,256], log=bool)
#plt.hist(img.ravel())
#plt.show()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/empty_daylight-' + img_name, dpi=150)
plt.close()


img_name = 'ipcam_2018-06-18_09.59.45.246.jpg'
img = cv2.imread(path + img_name,1)
#cv2.imshow(path + 'ipcam_2018-06-14_11.58.45.536.jpg',img)
#plt.hist(img.ravel(),256,[0,256], log=bool)
#plt.hist(img.ravel())
#plt.show()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/empty_daylight-' + img_name, dpi=150)
plt.close()

#empty artifical
img_name = 'ipcam_2018-06-16_01.19.08.337.jpg'
img = cv2.imread(path + img_name,1)
#cv2.imshow(path + 'ipcam_2018-06-14_11.58.45.536.jpg',img)
#plt.hist(img.ravel(),256,[0,256], log=bool)
#plt.show()color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/empty_artifical-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-16_21.49.39.110.jpg'
img = cv2.imread(path + img_name,1)
#cv2.imshow(path + 'ipcam_2018-06-14_11.58.45.536.jpg',img)
#plt.hist(img.ravel(),256,[0,256], log=bool)
#plt.show()color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/empty_artifical-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-17_19.02.33.893.jpg'
img = cv2.imread(path + img_name,1)
#cv2.imshow(path + 'ipcam_2018-06-14_11.58.45.536.jpg',img)
#plt.hist(img.ravel(),256,[0,256], log=bool)
#plt.show()color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/empty_artifical-' + img_name, dpi=150)
plt.close()

#empty ambient
img_name = 'ipcam_2018-06-13_21.36.14.918.jpg'
img = cv2.imread(path + img_name,1)
#cv2.imshow(path + 'ipcam_2018-06-14_11.58.45.536.jpg',img)
#plt.hist(img.ravel(),256,[0,256], log=bool)
#plt.show()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/empty_ambient-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-15_20.05.47.232.jpg'
img = cv2.imread(path + img_name,1)
#cv2.imshow(path + 'ipcam_2018-06-14_11.58.45.536.jpg',img)
#plt.hist(img.ravel(),256,[0,256], log=bool)
#plt.show()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/empty_ambient-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-16_06.36.13.788.jpg'
img = cv2.imread(path + img_name,1)
#cv2.imshow(path + 'ipcam_2018-06-14_11.58.45.536.jpg',img)
#plt.hist(img.ravel(),256,[0,256], log=bool)
#plt.show()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/empty_ambient-' + img_name, dpi=150)
plt.close()

path='../ipcam_images/train/intruder/'

#fullbody close night
img_name = 'ipcam_2018-06-13_22.28.40.469.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(close)_night-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-17_19.46.13.101.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(close)_night-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-17_23.00.00.651.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(close)_night-' + img_name, dpi=150)
plt.close()


#fullbody far night
img_name = 'ipcam_2018-06-13_22.24.39.938.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(far)_night-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-13_22.24.47.376.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(far)_night-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-17_22.59.12.354.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(far)_night-' + img_name, dpi=150)
plt.close()


#fullbody close daylight
img_name = 'ipcam_2018-06-15_17.01.43.245.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(close)_daylight-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-15_07.26.42.876.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(close)_daylight-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-15_19.41.28.470.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(close)_daylight-' + img_name, dpi=150)
plt.close()


#fullbody far daylight
img_name = 'ipcam_2018-06-16_08.59.53.249.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(far)_daylight-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-17_16.38.09.663.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(far)_daylight-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-16_18.41.23.842.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(far)_daylight-' + img_name, dpi=150)
plt.close()


#fullbody close artificial
img_name = 'ipcam_2018-06-15_17.01.43.245.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(close)_artificial-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-17_18.23.17.894.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(close)_artificial-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-18_07.15.22.052.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(close)_artificial-' + img_name, dpi=150)
plt.close()


#fullbody far artificial
img_name = 'ipcam_2018-06-14_06.52.50.093.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(far)_artificial-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-17_18.06.39.022.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(far)_artificial-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-16_18.41.23.842.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(far)_artificial-' + img_name, dpi=150)
plt.close()

#fullbody close ambient
img_name = 'ipcam_2018-06-16_19.01.54.452.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(close)_ambient-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-16_19.10.46.225.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(close)_ambient-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-17_15.40.28.684.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(close)_ambient-' + img_name, dpi=150)
plt.close()


#fullbody far artificial
img_name = 'ipcam_2018-06-16_19.00.39.246.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(far)_ambient-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-17_14.51.00.458.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(far)_ambient-' + img_name, dpi=150)
plt.close()

img_name = 'ipcam_2018-06-17_16.37.58.882.jpg'
img = cv2.imread(path + img_name,1)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig('output/fullbody_(far)_ambient-' + img_name, dpi=150)
plt.close()
