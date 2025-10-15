'''
pip install opencv-python
pip install cmake
pip install dlib
pip install face-recognition
pip install numpy
'''
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
import cv2
import numpy as np
import face_recognition as fr
import os
import pickle 

class SistemaReconhecimentoFacial:
    def __init__(self):
        self.encodings_conhecidos = []
        self.nomes_conhecidos = []
        self.carregar_dados_treinamento()

    def carregar_dados_treinamento(self):
        direc = os.path.dirname(os.path.abspath(__file__))
        pasta_treinamento = os.path.join(direc, 'pessoas_conhecidas')

        if not os.path.exists(pasta_treinamento):
            os.makedirs(pasta_treinamento)
            print(f"Pasta '{pasta_treinamento}' criada. Adicione fotos para o treinamento")
    
        for arquivo in os.listdir(pasta_treinamento):
            if arquivo.lower().endswith(('.jpg','.jpeg','.png')):
                caminho_imagem = os.path.join(pasta_treinamento, arquivo)

                imagem = fr.load_image_file(caminho_imagem)

                encodings = fr.face_encodings(imagem)

                if len(encodings) > 0:
                    encodings = encodings[0]
                    self.encodings_conhecidos.append(encodings)
                    nome = os.path.splitext(arquivo)[0]
                    self.nomes_conhecidos.append(nome)
                    print(f"'{nome}' adicionado no sistema")

                else:
                    print(f"Nenhum rosto detectado em '{arquivo}'")

    def pre_processamento(self, imagem):
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) #Converte a imagem para cinza
        equalizada = cv2.equalizeHist(cinza) # equalização de histograma para melhorar contraste
        suavizada = cv2.GaussianBlur(equalizada, (5,5), 0) #filtro Gaussiano para reduzir ruido

        return suavizada

    def detectar_rostos(self,imagem):
        #Usando face recognition (mais preciso)
        rostos_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        locais_rostos = fr.face_locations(rostos_rgb)

        #Usando Haar Cascades
        cinza = self.pre_processamento(imagem)
        classificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        rostos_haar = classificador.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

        locais_haar = []
        for (x, y, w, h) in rostos_haar:
            locais_haar.append((y, x + w, y + h, x))

        if len(locais_rostos) > 0:
            return locais_rostos
        else:
            return locais_haar
    
    def reconhecer_rosto(self, imagem):
        locais_rostos = self.detectar_rostos(imagem)

        if not locais_rostos:
            return []
        
        #obtem encoding
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        encoding_rostos = fr.face_encodings(imagem_rgb, locais_rostos)

        resultado = []

        for encoding_rosto, local_rosto in zip(encoding_rostos, locais_rostos):
            distancias = fr.face_distance(self.encodings_conhecidos, encoding_rosto)

            if len(distancias) > 0:
                melhor_indice = np.argmin(distancias)
                melhor_distancia = distancias[melhor_indice]

                threshold = 0.6

                if melhor_distancia < threshold:
                    nome = self.nomes_conhecidos[melhor_indice]
                    confianca = (1 - melhor_distancia) * 100
                else:
                    nome = "Desconhecido"
                    confianca = (1 - melhor_distancia) * 100
                
                resultado.append({
                    'local': local_rosto,
                    'nome': nome,
                    'confianca': confianca,
                    'encoding': encoding_rosto
                })
        return resultado
    
    def teste_com_imagem(self, caminho_imagem):

        imagem = cv2.imread(caminho_imagem)
        if imagem is None:
            print("Erro ao carregar a imagem")
            return
        
        resultados = self.reconhecer_rosto(imagem)

        for resultado in resultados:
            top, right, bottom, left = resultado['local']
            nome = resultado['nome']
            confianca = resultado['confianca']

            print(f"Rosto detectado: {nome} \n"+
                  "Confiança: {confianca: .1f}% ")
                  
        cv2.rectangle(imagem, (left, top), (right, bottom), (0,255,0),2)
        cv2.putText(imagem, f"{nome} ({confianca: .1f}%)",
                    (left,top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
        
        cv2.imshow('Resultado', imagem)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def arquivos_carregados(self):
        print("\n------Arquivos no sistema--------")
        for i, arquivo in enumerate(self.nomes_conhecidos):
            print(f"{i+1},{arquivo}")
        print(f"Total: {len(self.nomes_conhecidos)} arquivos")

def main():
    sistema = SistemaReconhecimentoFacial()
    arquivo = input("Nome do arquivo para o reconhecimento facial:\n")

    direc = os.path.dirname(os.path.abspath(__file__))
    caminho_imagem = os.path.join(direc,'pessoas', arquivo)
    sistema.teste_com_imagem(caminho_imagem)

if __name__ == "__main__":
    main()