import cv2 as cv

#Esta funcion recive todos lo datos de las etapas de Deteccion,Emparejamiento, y almacena las imagenes
#resultantes de cada proceso por separado, luego genera un string con el nombre del detector y el emparejador 
#que se utilizo.
def save(img3,emparejador,detector,norma,tag):

    if emparejador == 11 or emparejador == 12:
        emp="bf"
        if detector ==1:
            cv.imwrite(str(tag) + "_GFTT_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector == 2:
            cv.imwrite(str(tag) + "_FAST_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector == 3:
            cv.imwrite(str(tag) + "_BRIEF_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==4:
            cv.imwrite(str(tag) + "_ORB_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==5:
            cv.imwrite(str(tag) + "_AGAST_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==6:
            cv.imwrite(str(tag) + "_AKAZE_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==7:
            cv.imwrite(str(tag) + "_BRISK_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==8:
            cv.imwrite(str(tag) + "_KAZE_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==9:
            cv.imwrite(str(tag) + "_SIFT_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==10:
            cv.imwrite(str(tag) + "_SURF_" +  str(emp) + str(norma) +  "_.png",img3)
    if emparejador == 13 or emparejador == 14:
        emp="FL"
        if detector ==1:
            cv.imwrite(str(tag) + "_GFTT_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector == 2:
            cv.imwrite(str(tag) + "_FAST_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector == 3:
            cv.imwrite(str(tag) + "_BRIEF_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==4:
            cv.imwrite(str(tag) + "_ORB_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==5:
            cv.imwrite(str(tag) + "_AGAST_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==6:
            cv.imwrite(str(tag) + "_AKAZE_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==7:
            cv.imwrite(str(tag) + "_BRISK_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==8:
            cv.imwrite(str(tag) + "_KAZE_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==9:
            cv.imwrite(str(tag) + "_SIFT_" +  str(emp) + str(norma) +  "_.png",img3)
        elif detector ==10:
            cv.imwrite(str(tag) + "_SURF_" +  str(emp) + str(norma) +  "_.png",img3)