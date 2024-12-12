#Importar Librerias
import cv2
import os

#Importar HandFollowing
import HandsFollowing as HandsFollowing

#Crea carpeta
name = "Letter_A"
path = "X:/SignRecognition/data"
folder = path + "/"+ name

#Si no está creada
if not os.path.exists(folder):
    print("Carpeta creada: ", folder)
    os.makedirs(folder)

#Cam Reading
cap = cv2.VideoCapture(0)

#Cambiar resolución
cap.set(3, 1280)
cap.set(4, 720)
count = 0

#Declarar detector de manos
Detector = HandsFollowing.HandTracking(Confdetection=0.9)

while True:
    #Realizar lectura de captura
    ret, img = cap.read()

    #Extraer información de la mano
    img = Detector.findHands(img, draw= False)

    #Extraer posición de una sola mano
    lista1, bBox, hand = Detector.findPosition(img, DrawPoints=False, DrawBox=False, color=[0,255,0])

    #Si encontró mano
    if hand == 1:
        #Extraer info del recuadro
        xmin, ymin, xmax, ymax = bBox

        #Agregar margen al recuadro de reconocimiento
        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40

        #Recortar imagen de la mano
        imgCrop = img[ymin:ymax, xmin:xmax]

        imgCrop = cv2.resize(imgCrop, (640,640), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(folder + "/A_{}.jpg".format(count), imgCrop)
        count += 1

        cv2.imshow("Recorte", imgCrop)
        cv2.rectangle(img, (xmin, ymin), (xmax,  ymax),[255,0,0], 2)

    cv2.imshow("SIGN CAPTURE", img)
    t = cv2.waitKey(1)
    if t == 27 or count == 100   :
        break

cap.release()
cv2.destroyAllWindows(  )