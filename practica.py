import cv2 # IMPORTACIÓN DEL PAQUETE OPENCV
# CARGA DE PATRONES PARA EL ALGORITMO "HAAR CASCADE" PREDEFINIDOS EN EL PAQUETE OPENCV
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
imagenOriginal = cv2.imread('imagenprueba.jpg') # CARGA DE LA IMAGEN EN LA QUE SE DETECTARÁN ROSTROS
imagenCopia = imagenOriginal.copy() # SE COPIA LA IMAGEN PARA COLOCAR RECTÁNGULOS EN LAS CARAS IDENTIFICADAS
imagenGrises = cv2.cvtColor(imagenOriginal, cv2.COLOR_BGR2GRAY) # CONVIERTE A TONOS GRISES LA IMÁGEN A RECONOCER
# SE APLICA EL ALGORITMO DE CLASIFICACIÓN DE "HAAR CASCADE" EN LA IMAGEN EN TONOS DE GRISES
carasDetectadas = faceClassif.detectMultiScale(imagenGrises,
scaleFactor=1.1,
minNeighbors=5,
minSize=(30,30),
maxSize=(200,200))

contador = 0

for (x,y,ancho,alto) in carasDetectadas: # TOMA LAS COORDENADAS DE CADA ROSTRO DETECTADO
# PINTA UN RECTÁNGULO EN LAS COORDENADAS DE UNA CARA DETECTADA

    cv2.rectangle(imagenOriginal, (x,y),(x+ancho,y+alto),(128,0,255),2)

# RECORTA UNA CARA DETECTADA Y GUARDALA EN LA VARIABLE imagenDeUnRostro
imagenDeUnRostro = imagenCopia[y:y+alto,x:x+ancho]
# AJUSTA LA CARA DETECTADA AL TAMAÑO DE 150x150 PIXELES
imagenDeUnRostro = cv2.resize(imagenDeUnRostro,(150,150), interpolation=cv2.INTER_CUBIC)
# GUARDA EN UN ARCHIVO CON NOMBRE rostro_CONTADOR DE LA CARA DETECTADA
cv2.imwrite('rostro_{}.jpg'.format(contador),imagenDeUnRostro)
# INCREMENTA UNO AL CONTADOR DE CARAS DETECTADAS
contador = contador + 1
# MUESTRA LA IMAGEN DE LA CARA DETECTADA EN ESTA ITERACIÓN
cv2.imshow('rostro',imagenDeUnRostro)
# MUESTRA LA IMAGEN ORIGINAL CON UN RECTÁNGULO EN LA CARA DETECTADA EN ESTA ITERACIÓN
cv2.imshow('imagen',imagenOriginal)
# SOLICITA QUE SE PULSE ENTER EN LA PANTALLA DE LA IMAGEN ORIGINAL PARA VER OTRA CARA SI ES QUE LA HAY
print("Pulsa la tecla ENTER>")
cv2.waitKey(0)
# LIBERA LA MEMORIA DE LAS IMAGENES MOSTRADAS
cv2.destroyAllWindows()