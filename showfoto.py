import cv2
import os
import numpy as np


caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
#print(caminhos)
faces = []
ids = []

for caminhoImagem in caminhos:
    imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
    cv2.imshow("Faces", imagemFace)
    print(np.average(imagemFace))
    if cv2.waitKey(0) == ord('q'):
        break

