import numpy as np

from sift import *
from agast import *
from akaze import *
from brief import *
from brisk import *
from kaze import *
from orb import *
from surf import *
from gftt import *
from fast import *
from menu_emparejamiento import *
import tkinter as tk

from tkinter import filedialog
root=tk.Tk()
root.withdraw()



print("****************************************************************************")
print("******Deteccion, descripcion y Emparejamiento de Rasgos en OpenCV***********")
print("****************************************************************************")

print("Ingrese las imagenes a analizar, seleccione una opcion:")
print("Opcion 1: Imagenes preestablecidas  presione --> 'd'")
print("Opcion 2: Ingresar el nombre de ambas imagenes de prueba --> 'n'")


opcion=input("Ingrese una opcion:__")
if opcion == 'd':
    print("Selecciono usar imagenes preestablecidas \n")
    nombre1=''
    nombre2=''
if opcion == 'db':
    print("Selecciono usar imagenes preestablecidas \n")
    nombre1=''
    nombre2=''
if opcion == 'n':
    nombre1=filedialog.askopenfilename()
    nombre2=filedialog.askopenfilename()
    #nombre1=input("Ingrese el nombre de la imagen 1:__")
    #nombre2=input("Ingrese el nombre de la imagen 2:__")
    print(nombre1,nombre2)
print("Seleccione un metodo de Deteccion de Rasgos")
print("1-Good Features to Track     2-FAST      3-BRIEF     4-ORB       5-AGAST")
print("6-AKAZE                      7-BRISK     8-KAZE      9-SIFT      10-SURF")
detector = int(input("Detector:__"))

print("Seleccione un metodo de emparejamiento")
print("a-Brute-force               b-Flann")
emparejador = input("Emparejador:__")

if emparejador == 'a':

    if detector == 1 :
        norma=''
        print("Detector:Good Features to Track")
        print("Emparejador:Brute_force")
        print("Norma: Hamming distance")
        gftt(detector,11,opcion,nombre1,nombre2,norma)
    if detector == 2:
        norma=''
        print("Detector:FAST")
        print("Emparejador:Brute_force")
        print("Norma: Hamming distance")
        fast(detector,11,opcion,nombre1,nombre2,norma)
    if detector == 3:
        norma=''
        print("Detector:BRIEF")
        print("Emparejador:Brute_force")
        print("Norma: Hamming distance")
        brief(detector,11,opcion,nombre1,nombre2,norma)
    if detector == 4:
        norma=''
        print("Detector:ORB")
        print("Emparejador:Brute_force")
        print("Norma: Hamming distance")
        orb(detector,11,opcion,nombre1,nombre2,norma)
    if detector == 5:
        norma=''
        print("Detector:AGAST")
        print("Emparejador:Brute_force")    
        print("Norma: Hamming distance")
        agast(detector,11,opcion,nombre1,nombre2,norma)
    if detector == 6:
        norma=''
        print("Detector:AKAZE")
        print("Emparejador:Brute_force")
        print("Norma: Hamming distance")
        akaze(detector,11,opcion,nombre1,nombre2,norma)
    if detector == 7:
        norma=''
        print("Detector:BRISK")
        print("Emparejador:Brute_force")
        print("Norma: Hamming distance")
        brisk(detector,11,opcion,nombre1,nombre2,norma)
    if detector == 8:
        print("Detector:KAZE")
        print("Emparejador:Brute_force")    
        norma=int(input("Seleccione norma para medicion de distancia:\n\
                 1) Norm_L1   2)Norm_L2 -- Seleccion:__ "))
        kaze(detector,12,opcion,nombre1,nombre2,norma)
    if detector == 9:
        print("Detector:SIFT")
        print("Emparejador:Brute_force")    
        norma=int(input("Seleccione norma para medicion de distancia:\n\
                 1) Norm_L1   2)Norm_L2 -- Seleccion:__ "))
        sift(detector,12,opcion,nombre1,nombre2,norma)
    if detector == 10:
        print("Detector:SRF")    
        print("Emparejador:Brute_force")    
        norma=int(input("Seleccione norma para medicion de distancia:\n\
                 1) Norm_L1   2)Norm_L2 -- Seleccion:__ "))
        surf(detector,12,opcion,nombre1,nombre2,norma)
        

if emparejador == 'b':
    norma=''
    if detector == 1 :
        print("Detector:Good Features to Track")
        print("Emparejador:FLANN")    
        gftt(detector,13,opcion,nombre1,nombre2,norma)
    if detector == 2:
        print("Detector:FAST")
        print("Emparejador:FLANN")    
        fast(detector,13,opcion,nombre1,nombre2,norma)
    if detector == 3:
        print("Detector:BRIEF")
        print("Emparejador:FLANN")    
        brief(detector,13,opcion,nombre1,nombre2,norma)
    if detector == 4:
        print("Detector:ORB")
        print("Emparejador:FLANN")    
        orb(detector,13,opcion,nombre1,nombre2,norma)
    if detector == 5:
        print("Detector:AGAST")
        print("Emparejador:FLANN")    
        agast(detector,13,opcion,nombre1,nombre2,norma)
    if detector == 6:
        print("Detector:AKAZE")
        print("Emparejador:FLANN")    
        akaze(detector,13,opcion,nombre1,nombre2,norma)
    if detector == 7:
        print("Detector:BRISK")
        print("Emparejador:FLANN")    
        brisk(detector,13,opcion,nombre1,nombre2,norma)
    if detector == 8:
        print("Detector:KAZE")
        print("Emparejador:FLANN")    
        kaze(detector,14,opcion,nombre1,nombre2,norma)
    if detector == 9:
        print("Detector:SIFT")
        print("Emparejador:FLANN")    
        sift(detector,14,opcion,nombre1,nombre2,norma)
    if detector == 10:
        print("Detector:SURF")    
        print("Emparejador:FLANN")    
        surf(detector,14,opcion,nombre1,nombre2,norma)
        
