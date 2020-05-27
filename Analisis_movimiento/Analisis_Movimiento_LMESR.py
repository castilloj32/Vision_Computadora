from __future__ import print_function
from __future__ import division
from math import atan2, cos, sin, sqrt, pi
import cv2 as cv
import math 
import numpy as np
import argparse


print("Este programa Modela el fondo de un video, por medio del promedio \n\
      de cuadros de video, utilizando el metodo de minimos cuadrados")

print("Ingrese el video a analizar, seleccione una opcion:")
print("Opcion 1: video preestablecido,  presione --> '1'")
print("Opcion 2: Utilizar la camara de la computadora, presione --> '2'")
print("Opcion 3: Ingresar el nombre del video, presione--> '3'")


opcion=int(input("Ingrese una opcion:__"))

if opcion == 1:
    print ("selecciono opcion 1")
if opcion == 2:
    nombre=input("Selecciono la opcion 3 \n, ingrese el nombre del video con extension --->")
    input("presione una tecla para continuar")
input("presione una tecla para continuar")

################################################################################################################################
############PCA analisis, calculo gradiente y orientacion#######################################################################
################################################################################################################################
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    ## visualizacion
    angle = atan2(p[1] - q[1], p[0] - q[0]) # Angulo en radianes
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Se indica el tamaño de la flecha, por un valor de escala
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    # Se crea el gancho de la flecha
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)


def getOrientation(pts, img):
    # Se crea un bufer usado durante el analisis PCA
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]

    # Se realiza analisis PCA
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    # Se almacena el centro del objeto
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    # Se dibujan los componentes principales
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)

    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientacion en radianes

    return angle


################################################################################################################################
############Calculo fondo#######################################################################################################
################################################################################################################################
#Load Video
if opcion == 1:
    capture = cv.VideoCapture('vtest.avi')
    
if opcion == 2:
    capture = cv.VideoCapture(nombre)
    
if not capture.isOpened:
    print('Unable to open: ' )
    exit(0)

#Obtener Ancho,altura y #cuadros del video
Vwidth = capture.get(3)
Vheight = capture.get(4)
fps = round(capture.get(5),0)

print(str(Vwidth) + 'x' + str(Vheight) + ' , ' + str(fps) + 'fps')

#Obtener el numero total de cuadros
TotalFrames =  int(capture.get(cv.CAP_PROP_FRAME_COUNT))
print("Total Frames: " ,TotalFrames )


while True:
#Captura los cuadros del video
    ret_ave, frame_ave = capture.read()
    
#Se obtiene el numero del cuadro actual 
    Current_Frame_POS = int( capture.get(1) )
    Learnstr = "Learning: " + str(Current_Frame_POS) + "/" + str(TotalFrames) + " " + str(round(Current_Frame_POS/ TotalFrames*100,1)) +'%' 
    print(Learnstr)
    if frame_ave is None:
        break

#Se inicia el calculo del promedio de pixeles en cada cuadro
    if Current_Frame_POS == 1:
        avg = np.float32(frame_ave)

#Calculo del promedio aculumado entre cada cuadro para cada pixel.
    cv.accumulateWeighted(frame_ave, avg, 0.005) 
    result = cv.convertScaleAbs(avg)
#Almacenamiento de el valor minimo de cada promedio calculado en la variable
#"result_min"
    np.array(result)
    result_min = np.min(result)
    result_bg=result-result_min
    cv.imshow('Imagen original', frame_ave)
    cv.imshow('Imagen fondo',result_bg)



################################################################################################################################
############Analisis objetos en movimiento (Segmentacion)#######################################################################
################################################################################################################################

if opcion == 1:
    capture = cv.VideoCapture('vtest.avi')

if opcion == 2:
    capture = cv.VideoCapture(nombre)

if not capture.isOpened:
    print('Unable to open: ' )
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    #Se convierte la imagen en cada cuadra a escala de grises, y formato binario.
    fgMask=frame-result_bg
    fgMask = cv.cvtColor(fgMask, cv.COLOR_RGB2GRAY)
    fgMask = cv.GaussianBlur(fgMask,(5,5),0)
    _, fgMask = cv.threshold(fgMask, 70, 150, cv.THRESH_BINARY_INV)
    cv.imshow('Eliminacion del fondo',fgMask)
    #Se procesan la imagen de los objetos segmentados mediante
    #funciones de apertura y cerradura.
    kernel = np.ones((5,5),np.uint8)
    fgMask_bin = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
    fgMask_bin = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel)

    # Encuentra los contornos en la imagen binaria.
    contours, _ = cv.findContours(fgMask_bin, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    
    for i, c in enumerate(contours):
        # Calcula el area de cada contorno
        area = cv.contourArea(c)
        # Ignora contornos que son demasiado pequeños o demasiado grandes
        if area < 1e2 or 1e5 < area:
            continue
        # Dibuja cada contorno
        cv.drawContours(frame, contours, i, (0, 0, 255), 2)
        # Encuentra la orientacion de cada objeto
        getOrientation(c, frame)
    #Muestra el video de entrada he incluye los contorno detectados.
    cv.imshow('output',frame)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        capture.release()
        cv.destroyAllWindows()        
        break
