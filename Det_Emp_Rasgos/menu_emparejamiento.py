import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from save_file import *


def menu_emparejamiento(emparejador,kp1,des1,kp2,des2,img1,img2,norma,detector):

    
        #Almacena los valores de cada slider en variables separadas y las dirige a el
        #proceso de emparejamiento correcto, ya sea con punto flotanto o binario.
    def slider_brute_force_binary(val_slider):
        kn_matches=cv.getTrackbarPos(title_trackbar2, window_name)
        brute_force_binary(kn_matches)
    def slider_brute_force_flotant(val_slider):
        kn_matches=cv.getTrackbarPos(title_trackbar, window_name)
        brute_force_flotant(kn_matches)
    def slider_flann_binary(val_slider):
        ratio_thresh=cv.getTrackbarPos(title_trackbar, window_name)
        search_times=cv.getTrackbarPos(title_trackbar3, window_name)
        flann_binary(ratio_thresh,search_times)
    def slider_flann_flotant(val_slider):
        ratio_thresh=cv.getTrackbarPos(title_trackbar, window_name)
        search_times=cv.getTrackbarPos(title_trackbar3, window_name)
        flann_flotant(ratio_thresh,search_times)
        
    
        
        
    def brute_force_binary(kn_matches):    
        
        #Inicia Emparejador BF con medicion de distancia Hamming
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        #Encuentra los emparejamientos entre los descriptores
        matches = bf.match(des1,des2)
        #Ordena los emparejamientos hallados en orden ascendente
        #Los Mas cercanos se ubican primero
        matches = sorted(matches, key = lambda x:x.distance)
        
        #Conserva solo los mejores n emparejamientos indicados por "kn_matches" 
        img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:kn_matches],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        #Muestra y almacena la imagen con los emparejamientos
        img3 = cv.resize(img3,(1000,500))
        cv.imshow(window_name,img3)
        save(img3,emparejador,detector,norma,tag)
        
        
        
        
    def brute_force_flotant(kn_matches):    
        
        
        if norma == 1 : 
            #Inicia Emparejador BF con medicion de distancia Norm_L1
            bf = cv.BFMatcher(cv.NORM_L1,crossCheck=True)
            #Encuentra los emparejamientos entre los descriptores
            matches = bf.match(des1,des2)
            #Ordena los emparejamientos hallados en orden ascendente
            #Los Mas cercanos se ubican primero
            matches = sorted(matches, key = lambda x:x.distance)
            #Conserva solo los mejores n emparejamientos indicados por "kn_matches" 
            img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:kn_matches],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #Muestra y almacena la imagen con los emparejamientos
            img3 = cv.resize(img3,(1000,500))
            cv.imshow(window_name,img3)
            save(img3,emparejador,detector,norma,tag)
            
        if norma == 2 :
            #Inicia Emparejador BF con medicion de distancia Norm_L2
            bf = cv.BFMatcher(cv.NORM_L2,crossCheck=True)
            #Encuentra los emparejamientos entre los descriptores
            matches = bf.match(des1,des2)
            #Ordena los emparejamientos hallados en orden ascendente
            #Los mas cercanos se ubican primero
            matches = sorted(matches, key = lambda x:x.distance)
            #Conserva solo los mejores n emparejamientos indicados por "kn_matches" 
            img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:kn_matches],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #Muestra y almacena la imagen con los emparejamientos
            img3 = cv.resize(img3,(1000,500))
            cv.imshow(window_name,img3)
            save(img3,emparejador,detector,norma,tag)

    def flann_binary(ratio_thresh,search_times):
        
        ratio_thresh=ratio_thresh/10
        
        #Inicia metodo Flann con LSH
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, 
                       key_size = 12,     
                       multi_probe_level = 1) 
        search_params = dict(checks=search_times)   
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        
        #Evalua y conserva solo los emparejamientos 
        #que se encuentren dentro de un radio de distancia.
        matchesMask = [[0,0] for i in range(len(matches))]
        for i,(m,n) in enumerate(matches):
            if m.distance < ratio_thresh*n.distance:
                matchesMask[i]=[1,0]
        #Almacena los emparejamientos validos encontrados
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = cv.DrawMatchesFlags_DEFAULT)
        
        #Muestra y almacena la imagen con los emparejamientos
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        img3 = cv.resize(img3,(1000,500))
        cv.imshow(window_name,img3)
        save(img3,emparejador,detector,norma,tag)
    
    
    
    def flann_flotant(ratio_thresh,search_times):
        
        ratio_thresh=ratio_thresh/10
        
        #Inicia metodo Flann con KDTREE=5
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=search_times)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
    
    
        #Evalua y conserva solo los emparejamientos 
        #que se encuentren dentro de un radio de distancia.
        matchesMask = [[0,0] for i in range(len(matches))]
        for i,(m,n) in enumerate(matches):
            if m.distance < ratio_thresh*n.distance:
                matchesMask[i]=[1,0]
        
        #Almacena los emparejamientos validos encontrados
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = cv.DrawMatchesFlags_DEFAULT)
        
        #Muestra y almacena la imagen con los emparejamientos
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        img3 = cv.resize(img3,(1000,500))
        cv.imshow(window_name,img3)
        save(img3,emparejador,detector,norma,tag)
    
        
        
        #Crea los sliders que se usan en cada proceso de emparejamiento para variar el numero de
        #emparejamientos a mostrar, el numero de busquedas repetidas entre arboles, y el radio de distancia entre
        #los emparejamientos mas cercanos.
    if emparejador == 11:
        title_trackbar2="# Matches"
        window_name="Emparejamiento de rasgos"
        cv.namedWindow(window_name)    
        cv.resizeWindow(window_name,500,400)
        cv.createTrackbar(title_trackbar2, window_name , 10, 1000, slider_brute_force_binary)
        brute_force_binary(10)
        cv.waitKey()
        
    if emparejador == 12:
        title_trackbar="# Matches"
        window_name="Emparejamiento de rasgos"
        cv.namedWindow(window_name)    
        cv.resizeWindow(window_name,500,400)
        cv.createTrackbar(title_trackbar, window_name , 100, 1000, slider_brute_force_flotant)
        brute_force_flotant(100)
        cv.waitKey()
        
    if emparejador == 13:
        title_trackbar="Radio"
        title_trackbar3="Busquedas"
        window_name="Emparejamiento rasgos"
        cv.namedWindow(window_name)    
        cv.resizeWindow(window_name,500,400)
        cv.createTrackbar(title_trackbar, window_name , 9, 10, slider_flann_binary)
        cv.createTrackbar(title_trackbar3, window_name , 10, 200, slider_flann_binary)
        flann_binary(9,10)
        cv.waitKey()
        
    if emparejador == 14:
        title_trackbar="Radio"
        title_trackbar3="Busquedas"
        window_name="Emparejamiento rasgos"
        cv.namedWindow(window_name)
        cv.resizeWindow(window_name,500,400)
        cv.createTrackbar(title_trackbar, window_name , 9, 10, slider_flann_flotant)
        cv.createTrackbar(title_trackbar3, window_name ,100, 200, slider_flann_flotant)
        flann_flotant(9,100)
        cv.waitKey()
            
tag='Mtch'            