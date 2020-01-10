import cv2
import numpy as np

classificadorFace = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
#classificadorFaceOLho = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostras = 25
id = input('Digite seu Identificador: ')
largura, altura = 220, 220
print('Capturando Fotos...')

while True:
    conected, frame = camera.read()

    imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorFace.detectMultiScale(imagemCinza,scaleFactor=1.5,minSize=(150,150))

    for (x, y , l, a) in facesDetectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
        if cv2.waitKey(1)  == ord('q'):
            if np.average(imagemCinza) > 110:
                imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                cv2.imwrite('fotos/pessoa.' + str(id) + '.' + str(amostra) + '.jpg', imagemFace)
                print('[foto ' + str(amostra) + ' capturada com sucesso]')
                amostra += 1

    #print(np.average(imagemCinza))
    cv2.imshow('Face', frame)
    cv2.waitKey(1)
    if (amostra >= numeroAmostras + 1):
        break
print('Faces capturadas com sucesso')
camera.release()
cv2.destroyAllWindows()