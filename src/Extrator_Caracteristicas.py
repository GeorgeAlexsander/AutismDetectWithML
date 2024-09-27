# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:24:25 2024

@author: georg
"""

import cv2
print(cv2.__version__)

import numpy as np
import csv

# used for accessing url to download files
import urllib.request as urlreq

# used to access local directory
import os

# used to plot our images
import matplotlib.pyplot as plt

# used to change image size
from pylab import rcParams

# save picture's url in pics_url variable
# pics_url = "data:image/jpeg;base64,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"
pics_url = "https://www.iol.pt/multimedia/oratvi/multimedia/imagem/id/5693e19d0cf29f14c410ab92/600"
# save picture's name as pic
pic = "image.jpg"

# chech if picture is in working directory
if (pic in os.listdir(os.curdir)):
    print("Picture exists")
else:
    # download picture from url and save locally as image.jpg
    urlreq.urlretrieve(pics_url, pic)
    print("Picture downloaded")
# read image with openCV

image = cv2.imread(pic)

# convert image to RGB colour
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot image with matplotlib package
plt.imshow(image_rgb)

# Obter dimensões da imagem
height, width, _ = image_rgb.shape

# Definir proporções para o recorte (exemplo: 10% da largura e 20% da altura)

image_cropped = image_rgb.copy()

# Criar uma cópia da imagem recortada para uso posterior
image_template = image_cropped.copy()

# Mostrar a imagem recortada
plt.imshow(image_cropped)
plt.axis("off")
plt.show()

# Mostrar a imagem recortada
plt.imshow(image_cropped)

plt.axis("off")
plt.imshow(image_cropped)

# convert image to Grayscale
image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

# remove axes
plt.axis("off")
plt.imshow(image_gray, cmap = "gray")

# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"

# chech if file is in working directory
if (haarcascade in os.listdir(os.curdir)):
    print("File exists")
else:
    # download file from url and save locally as haarcascade_frontalface_alt2.xml
    urlreq.urlretrieve(haarcascade_url, haarcascade)
    print("File downloaded")
    
# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier(haarcascade)

# Detect faces using the haarcascade classifier on the "grayscale image"
faces = detector.detectMultiScale(image_gray)

# Print coordinates of detected faces
print("Faces:\n", faces)    

for face in faces:
#     save the coordinates in x, y, w, d variables
    (x,y,w,d) = face
    # Draw a white coloured rectangle around each face using the face's coordinates
    # on the "image_template" with the thickness of 2
    cv2.rectangle(image_template,(x,y),(x+w, y+d),(255, 255, 255), 2)

plt.axis("off")
plt.imshow(image_template)
plt.title('Face Detection')

# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "LFBmodel.yaml"

# check if file is in working directory
if (LBFmodel in os.listdir(os.curdir)):
    print("File exists")
else:
    # download picture from url and save locally as lbfmodel.yaml
    urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    print("File downloaded")
    
# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

# Detect landmarks on "image_gray"
_, landmarks = landmark_detector.fit(image_gray, faces)

# print coordinates of detected landmarks
print("landmarks LBF\n", landmarks)

# Criar uma imagem branca com as mesmas dimensões de image_cropped
height, width, _ = image_cropped.shape
image_blank = np.ones((height, width, 3), dtype=np.uint8) * 255  # Multiplicando por 255 para ter fundo branco

# Plotar os marcos na nova imagem branca
for landmark in landmarks:
    for i, (x, y) in enumerate(landmark[0]):
        # Exibir os marcos na imagem "image_blank"
        cv2.circle(image_blank, (int(x), int(y)), 1, (255, 0, 0), 2)
        
        # Adicionar o número do ponto logo acima dele
        cv2.putText(image_blank, str(i+1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

# Exibir ou salvar a imagem resultante
cv2.imshow("Landmarks on White Background", image_blank)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Abrir um arquivo CSV para salvar
with open('landmarks.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    for landmark in landmarks:
        # Aplanar o array de marcos (x1, y1, x2, y2, ..., x68, y68)
        flattened_landmark = landmark.flatten()
        # Escrever a linha no arquivo CSV
        writer.writerow(flattened_landmark)
