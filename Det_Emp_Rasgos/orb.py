import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from menu_emparejamiento import *
from save_file import *

def orb(detector,emparejador,opcion,nombre1,nombre2,norma):
    
    #Lee las imagenes a analizar
    if opcion == 'd':
        img1= cv.imread('tpic5.png',cv.IMREAD_GRAYSCALE)          
        img2 = cv.imread('tpic5_flipped.png',cv.IMREAD_GRAYSCALE) 
    
    if opcion == 'db':
        img1= cv.imread('thome.jpg',cv.IMREAD_GRAYSCALE)          
        img2 = cv.imread('thome_escale.jpg',cv.IMREAD_GRAYSCALE)  
    
    if opcion == 'dc':
        img1= cv.imread('tgrafizq.png',cv.IMREAD_GRAYSCALE)          
        img2 = cv.imread('tgrafder.png',cv.IMREAD_GRAYSCALE) 
        
    
    if opcion == 'n':
        img1= cv.imread(nombre1,cv.IMREAD_GRAYSCALE)          
        img2 = cv.imread(nombre2,cv.IMREAD_GRAYSCALE) 
        
    #Inicia el Detector y Descriptor 
    orb = cv.ORB_create()
    
    #Detecta Rasgos y Calcula el descriptor
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    #Muestra Rasgos y Guarda la imagen correspondiente
    imgx = cv.drawKeypoints(img1, kp1, None, color=(0,255,255))
    imgy = cv.drawKeypoints(img2, kp2, None, color=(0,255,255))
    window_namex="Rasgos Caracteristicos imagen 1"
    window_namey="Rasgos Caracteristicos imagen 2"
    cv.namedWindow(window_namex)
    cv.namedWindow(window_namey)
    cv.resizeWindow(window_namex,500,400)
    cv.resizeWindow(window_namey,500,400)
    cv.imshow(window_namex,imgx)
    cv.imshow(window_namey,imgy)
    save(imgx,emparejador,detector,norma,tag1)
    save(imgy,emparejador,detector,norma,tag2)
    
    #Envia los datos a la etapa de emparejamiento
    menu_emparejamiento(emparejador,kp1,des1,kp2,des2,img1,img2,norma,detector)
            
tag1="Det1"
tag2="Det2"        
        
    
if __name__ == "__main__":
    orb(detector,emparejador)
