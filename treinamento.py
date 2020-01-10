import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create(50, 17000 )
fisherface = cv2.face.FisherFaceRecognizer_create(2, 400)
lbph = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 50 )

def getImagemComId():
    caminhos = [os.path.join('fotos_treino', f) for f in os.listdir('fotos_treino')]
    #print(caminhos)
    faces = []
    ids = []

    for caminhoImagem in caminhos:
        print(caminhoImagem)
        imagem = cv2.imread(caminhoImagem)
        imagemFace = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)
        cv2.imshow("Faces", imagemFace)
        cv2.waitKey(30)
    return np.array(ids), faces

ids, faces = getImagemComId()
#print(ids)
#print(faces)

print("Treinando...")
eigenface.train(faces, ids)
eigenface.write('classificadores/classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadores/classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadores/classificadorLBPH.yml')

print("Treinamento realizado")
