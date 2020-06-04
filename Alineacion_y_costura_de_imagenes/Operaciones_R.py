import numpy as np
import imutils
import cv2
from past.builtins import xrange


#Deteccion de rasgos en entre cada par de mosaicos
def Detect_Feature_And_KeyPoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect and extract features from the image
    descriptors = cv2.xfeatures2d.SIFT_create()
    (Keypoints, features) = descriptors.detectAndCompute(image, None)

    Keypoints = np.float32([i.pt for i in Keypoints])
    return (Keypoints, features)


#Calcula coincidencias etre los rasgo de los mosaicos
def get_Allpossible_Match(featuresA,featuresB):
    match_instance = cv2.DescriptorMatcher_create("BruteForce")
    All_Matches = match_instance.knnMatch(featuresA, featuresB, 2)
    return All_Matches

#Se obtienen las coincidencias entre rasgos que son validos
#y se encuentran dentro de un radio de busqueda (Lowe_radio)
def All_validmatches(AllMatches,lowe_ratio):
    valid_matches = []
    
    for val in AllMatches:
        if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
            valid_matches.append((val[0].trainIdx, val[0].queryIdx))

    return valid_matches


#Calcula la matriz de homografia, tomando entrada las coordenadas de puntos coincidentes
#en ambas imagenes
def Compute_Homography(pointsA,pointsB,max_Threshold):
    (H, status) = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, max_Threshold)
    return (H,status)

#Llamadas a 3 process: Busqueda conincidencias, Busqueda coincidencias validas y Calculo de 
#matriz de homografia
def matchKeypoints( KeypointsA, KeypointsB, featuresA, featuresB,lowe_ratio, max_Threshold):

    AllMatches = get_Allpossible_Match(featuresA,featuresB);
    valid_matches = All_validmatches(AllMatches,lowe_ratio)

    if len(valid_matches) > 4:
        # construct the two sets of points
        pointsA = np.float32([KeypointsA[i] for (_,i) in valid_matches])
        pointsB = np.float32([KeypointsB[i] for (i,_) in valid_matches])

        (Homograpgy, status) = Compute_Homography(pointsA, pointsB, max_Threshold)

        return (valid_matches, Homograpgy, status)
    else:
        return None

#Se realiza una proyeccion perspectiva a la imagen a utilizando la 
#matriz de homografia calculada.
def getwarp_perspective(imageA,imageB,Homography):
    val = imageA.shape[1] + imageB.shape[1]
    result_image = cv2.warpPerspective(imageA, Homography, (val , imageA.shape[0]))

    return result_image

#Obtiene las dimensiones de una imagen, ancho y alto.
def get_image_dimension(image):
    (h,w) = image.shape[:2]
    return (h,w)
#crea matriz donde se almacenan los puntos coincidentes detectados
def get_points(imageA,imageB):

    (hA, wA) = get_image_dimension(imageA)
    (hB, wB) = get_image_dimension(imageB)
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    return vis

#Dibuja los puntos coincidentes detectados entre dos imagenes en la matriz vis.
def draw_Matches(imageA, imageB, KeypointsA, KeypointsB, matches, status):

    (hA,wA) = get_image_dimension(imageA)
    vis = get_points(imageA,imageB)

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        if s == 1:
            ptA = (int(KeypointsA[queryIdx][0]), int(KeypointsA[queryIdx][1]))
            ptB = (int(KeypointsB[trainIdx][0]) + wA, int(KeypointsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    return vis

#Crea degradado e el borde de unione de dos imagenes que se unen
def blending(A,B,Yv):
    # Genera piramide guasiana para imagen A
    G = A.copy()
    gpA = [G]
    for i in xrange(6):
        G = cv2.pyrDown(gpA[i])
        gpA.append(G)
        
    # Genera piramide laplaciana para imagen A
    lpA = [gpA[5]]
    for i in xrange(5,0,-1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        GE = cv2.pyrUp(gpA[i], dstsize = size)
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)


    # Genera piramide gausiana para imagen B
    G = B.copy()
    gpB = [G]
    for i in xrange(6):
        G = cv2.pyrDown(gpB[i])
        gpB.append(G)
    
    # Genera piramide laplaciana para imagen B
    lpB = [gpB[5]]
    for i in xrange(5,0,-1):
        size = (gpB[i-1].shape[1], gpB[i-1].shape[0])
        GE = cv2.pyrUp(gpB[i], dstsize = size)
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)


    #Concatena las imagenes creadas a partir de piramides laplacianas
    #he indica las cordendas de union en la variable Yv.
    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape
        y1=Yv
        ls = np.hstack((la[:,0:y1], lb[:,y1:]))
        LS.append(ls)
        
    #Reconstruye La imagen a partir de las piramides calculadas y
    #la almacena en la variable ls_
    ls_ = LS[0]
    for i in xrange(1,6):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize = size)
        ls_ = cv2.add(ls_, LS[i])    
        

    #Muestra la mezcla directa de la imagenes
    y2=Yv
    real = np.hstack((A[:,:y2],B[:,y2:]))
    cv2.imwrite('a-Pyramid_blending.jpg',ls_)
    cv2.imwrite('a-Direct_blending.jpg',real)
    return ls_

#Se realiza la costura de imagenes, para eso es necesario llamar a procesos: deteccion y emparejamiento
#de rasgos, proyeccion perspectiva y degradado de imagenes.    
def image_stitch(images, match_status,x,opcion,opcion2):
    lowe_ratio=0.75
    max_Threshold=4.0

    (imageB, imageA) = images
    (KeypointsA, features_of_A) = Detect_Feature_And_KeyPoints(imageA)
    (KeypointsB, features_of_B) = Detect_Feature_And_KeyPoints(imageB)

    #Llama a funcion para obtener emparejamientos de rasgos
    Values = matchKeypoints(KeypointsA, KeypointsB,features_of_A, features_of_B, lowe_ratio, max_Threshold)

    if Values is None:
        return None

    #Llama a funcion para  obtener la proyeccion perspectiva de imagen A
    (matches, Homography, status) = Values
    result_image =getwarp_perspective(imageA,imageB,Homography)
    cv2.imwrite("warped_image"+str(x)+'.jpg',result_image)
    
    #Concatena la imagen de referencia con la imagen proyectada y los une por
    #los extremos con relacion a sus puntos coincidentes (costura).
    result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    
    #Realiza el desvanecimiento (mezcla) de las imagenes concatenadas.
    if opcion==1 and x==opcion2:
        result_image=blending(result_image,result_image,imageB.shape[1])    
    
    if opcion==2:
        result_image=blending(result_image,result_image,imageB.shape[1])    
    
    
    if match_status:
        vis = draw_Matches(imageA, imageB, KeypointsA, KeypointsB, matches,status)

        return (result_image, vis)

    return result_image

def cropping(result,tipo_imagen): 
    if tipo_imagen == 1:
        # Se crea un rectangulo con bordes de 2 pixeles
        print("Reduciendo contornos ...")
        result = cv2.copyMakeBorder(result, 2, 2, 2, 2,
        	cv2.BORDER_CONSTANT, (0, 0, 0))
        
        #Convierte la imagen panorama a blanco y negro.
        
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        
        
        #Se buscan los contornos en la imagen, el contorno externo es un 
        #cuadrado
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
       
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        
        #se crea una imagen de mascara con la misma dimension
        #que la imagen binarizada, con valores cero
        mask = np.zeros(thresh.shape, dtype="uint8")
        #Se genera el rectangulo mas pequeño que se puede generar
        #verticalmente con los contornos detectados
        (x, y, w, h) = cv2.boundingRect(c)
        #Se dibuja el rectangulo (relleno de blanco, "-1") mas pequeño con los parametros 
        #identificados
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        #Se crean dos copias de la imagen mascara
        minRect = mask.copy()
        sub = mask.copy()
    
        #Se resta el rectangulo minimo (ceros) menos la imagen binarizada, 
        #esto vacia el interior del area de la imagen, y solo con serva el 
        #contorno.
        while cv2.countNonZero(sub) > 0:
        	#se erosiona rectangulo minimo, para facilitar 
            #la operacion de substraccion
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect,thresh)
    
        #Se detectan los contornos del rectangulo minimo
        cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        #Se detectan una vez mas el rectangulo mas pequeño que se puede calcular con
        #los puntos detectados
        (x3, y3, w3, h3) = cv2.boundingRect(c)
        
        #Muestra los limites de la imagen resultante, utilizando como limite
        #el rectangulo calculado.
    result = result[10:y3 + h3, 10:x3 + w3]        
    return result,x3,y3,w3,h3
        