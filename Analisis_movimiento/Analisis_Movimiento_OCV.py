from __future__ import print_function
from glob import glob
import cv2 as cv
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import itertools as it

print("Este programa Modela el fondo de un video, por medio de los metodos\n\
      de substraccion  disponibles en Opencv4")
print("Ingrese el video a analizar, seleccione una opcion:")
print("Opcion 1: video preestablecido,  presione --> '1'")
print("Opcion 2: Utilizar la camara de la computadora, presione --> '2'")
print("Opcion 3: Ingresar el nombre del video, presione--> '3'")
opcion=int(input("Ingrese una opcion:__"))

if opcion == 1:
    print ("selecciono opcion 1")
if opcion == 2:
    print ("selecciono opcion 1")
if opcion == 3:
    nombre=input("Selecciono la opcion 3 \n Ingrese el nombre del video con extension --->")
    input("presione una tecla para continuar")

metodo=int(input("Seleccine el metodo de substraccion de fondo: \n  4)MOG 5)MOG2 6)KNN 7)GMG 8)CNT --->>"))

if metodo==4 :
    print("selecciono el metodo MOG ")
if metodo==5 :
    print("selecciono el metodo MOG2")
if metodo==6 :
    print("selecciono el metodo KNN")
if metodo==7 :
    print("selecciono el metodo GMG")
if metodo==8 :
    print("selecciono el metodo CNT")
input("presione una tecla para continuar")


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


if metodo == 4:
    backSub =cv.bgsegm.createBackgroundSubtractorMOG()
if metodo == 5:
    backSub = cv.createBackgroundSubtractorMOG2()
if metodo == 6:
    backSub = cv.createBackgroundSubtractorKNN()
if metodo == 7:
    backSub =cv.bgsegm.createBackgroundSubtractorGMG()
    
if metodo == 8:
    backSub =cv.bgsegm.createBackgroundSubtractorCNT()

#Inicia captura de video
if opcion == 1:
    capture = cv.VideoCapture('vtest.avi')
if opcion == 2:
    capture = cv.VideoCapture(0)
if opcion == 3:
    capture = cv.VideoCapture(nombre)

if not capture.isOpened:
    print('Unable to open video ')
    exit(0)

#Se declara detector de personas
hog = cv.HOGDescriptor()
hog.setSVMDetector( cv.HOGDescriptor_getDefaultPeopleDetector() )


while True:
    ret, frame = capture.read()
    if frame is None:
        break
    #Se obtiene la imagen de fondo y se extrae de la imagen.
    fgMask = backSub.apply(frame)
    
    #Se aplican operaciones apertura y cerradura a imagen binaria con objetos segmentados.
    kernel = np.ones((5,5),np.uint8)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel)
    cv.imshow('Extraccion de fondo', fgMask)
    
    #Se imprime el numero actual de cuadro que esta siendo procesado
    #en la parte superior de el video resultante.
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    #Se identifican los contorno en la imagen sin fondo.
    contours, hierarchy = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #Se dibujan los contornos en el video de origen.
    image = cv.drawContours(frame, contours, -1, (0, 255, 0), 2)
        
    
    #Procesa la deteccion de perosonas y coloca un cuadro en cada posicion detectada
    found, w = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
        else:
            found_filtered.append(r)
    draw_detections(frame, found)
    draw_detections(frame, found_filtered, 3)
    print('%d (%d) found' % (len(found_filtered), len(found)))

    #Muestra imagen de origen con lo contornos detctados
    cv.imshow('Video origen', frame)
    

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        capture.release()
        cv.destroyAllWindows()
        break
