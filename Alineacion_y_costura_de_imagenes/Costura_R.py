from Operaciones_R import *
import imutils
from imutils import paths
import cv2

#########################################################################################
################################Costura de imagenes######################################
#########################################################################################
print("El siguiente programa realiza la costura de\n\
      una serie de imagenes que seran colocadas en la carpeta ./images/\n\
      solo renombre las imagenes de forma incremental con las letras del abecedario\n\
      siendo -a- la imagen izquierda superior -b- la imagen adyacente inmediata\n")
opcion=int(input("Indique si las imagenes se degradaran/mezclaran\n\
      para cada 2 imagenes o se mezcalara el total de imagenes\n\
      1-Mezclar c/Par de imagenes por separado\n\
      2-Mezclar todas de imagenes\n Opcion:_\n\
      Opcion:_"))  
if opcion==1:
    opcion2=int(input("Seleccione el numero del mosaico que desea desvanecer \n\
                      0-(#Max_imagenes-2)\n\
                      ejemplo: 0-8 \n #Mosaico:_"))
    print("Procesando...\n")
print("Procesando...\n")



#Carga de imagenes de folder ./images

imagePaths = sorted(list(paths.list_images('./images/')))
images = []

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	images.append(image)

no_of_images=len(images)
counter=no_of_images






#Se ajustan las dimenciones de las imagenes para que todas sean iguales

for i in range(no_of_images):
    images[i] = imutils.resize(images[i], width=1000)
    #pyramida(images[8])
for i in range(no_of_images):
    images[i] = imutils.resize(images[i], height=1000)
    
#################################################################################################################3

if no_of_images==2:
    x=no_of_images-2
    (result, matched_points) = image_stitch([images[0], images[1]],True,x,None,None)
    
else:
    #Se procesan las dos primeras imagenes que serviran como el mosaico de referencia
    x=no_of_images-2
    (result, matched_points) = image_stitch([images[no_of_images-2], images[no_of_images-1]], True,x,None,None)
    cv2.imwrite('Reference_image.jpg',images[0])
    cv2.imwrite('Matches'+str(x)+'.jpg',matched_points)
    cv2.imwrite('result'+str(x)+'.jpg',result)

    #Se procesan las imagenes adyacntes de forma secuencial y se van uniendo al mosaico de referencia
    for i in range(no_of_images - 2):
        x=no_of_images-i-3
        (result, matched_points) = image_stitch([images[x],result],True,x,None,None)
        cv2.imwrite('Matches'+str(x)+'.jpg',matched_points)
        cv2.imwrite('result'+str(x)+'.jpg',result)
tipo_imagen=1
result,x3,y3,w3,h3=cropping(result,tipo_imagen)
cv2.imwrite("Panorama.jpg",result)
print("El Panorama generado es (sin desvanecimiento): Panorama.jpg \n")
print("Generando Panorama con desvanecimiento\n")
#################################################################################################################3

if no_of_images==2:
    x=no_of_images-2
    (result, matched_points) = image_stitch([images[0], images[1]],True,x,opcion,opcion2)
    
else:
    #Se procesan las dos primeras imagenes que serviran como el mosaico de referencia
    x=no_of_images-2
    (result, matched_points) = image_stitch([images[no_of_images-2], images[no_of_images-1]], True,x,opcion,opcion2)
    cv2.imwrite('Reference_image.jpg',images[0])
    cv2.imwrite('Matches'+str(x)+'.jpg',matched_points)
    cv2.imwrite('result'+str(x)+'.jpg',result)

    #Se procesan las imagenes adyacntes de forma secuencial y se van uniendo al mosaico de referencia
    for i in range(no_of_images - 2):
        x=no_of_images-i-3
        (result, matched_points) = image_stitch([images[x],result],True,x,opcion,opcion2)
        cv2.imwrite('Matches'+str(x)+'.jpg',matched_points)
        cv2.imwrite('result'+str(x)+'.jpg',result)

#Se muestra el ortomosaico sin ningun recorte
#cv2.imwrite('No recortado.jpg',result)
#Se recortan las esquinas excedentes
#result=cropping(result,None)
result = result[10:y3 + h3, 10:x3 + w3]

#Se muestran el Panorama completo con bordes recortados
cv2.imwrite("Panorama_blurred.jpg",result)
print("Proceso terminado, El Panorama generado es (con desvanecimiento): Panorama_blurred.jpg \n")

cv2.waitKey(0)
cv2.destroyAllWindows()
