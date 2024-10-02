import unittest
import os
import cv2
import sys
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.abspath('../src'))

from feature_extraction  import (
    download_file,
    load_image,
    detect_faces,
    detect_landmarks,
    plot_landmarks,
    save_landmarks_to_csv,
    process_images_in_folder
)

class TestImageProcessingModule(unittest.TestCase):
    """Classe de testes para o módulo de processamento de imagens."""

    def test_download_file_valid(self):
        """Testa o download de um arquivo a partir de uma URL válida.

        Verifica se o arquivo é salvo corretamente no diretório atual.
        """
        # Teste se o arquivo é baixado corretamente de uma URL válida
        
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
        filename = "haarcascade_frontalface_alt2.xml"
        download_file(url, filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename) 

    def test_download_file_invalid_url(self):
        """Testa o download de um arquivo a partir de uma URL inválida.

        Verifica se uma exceção é levantada ao tentar baixar um arquivo de uma URL inválida.
        """        
        with self.assertRaises(Exception):  
            download_file("invalid_url", "testfile.txt")

    def test_load_image_valid(self):
        """Testa o carregamento de uma imagem a partir de um caminho válido.

        Verifica se a imagem é carregada corretamente e se as dimensões estão corretas.
        """
        
        # Caminho da imagem de teste
        image_path = 'test_images/test_image_0.jpg'
        # Cria uma imagem de teste
        image_size = (100, 100)
        image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        img = Image.fromarray(image)
        img.save(image_path)
        
        # Carrega a imagem
        loaded_image = load_image(image_path)
        # Verifica se a imagem foi carregada corretamente
        self.assertEqual(loaded_image.shape, (100, 100, 3))
        os.remove(image_path)

    def test_load_image_invalid_path(self):
        """Testa o carregamento de uma imagem a partir de um caminho inválido.

        Verifica se um FileNotFoundError é levantado ao tentar carregar uma imagem que não existe.
        """
        with self.assertRaises(FileNotFoundError):
            load_image('invalid_path.jpg')

    def test_detect_faces_valid(self):
        """Testa a detecção de faces em uma imagem com faces conhecidas.

        Verifica se as coordenadas das faces detectadas estão no formato esperado.
        """        
        
        image = load_image('test_images/test_face_valid_0.jpg')     
           
        faces = detect_faces(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), '../data/pretrained_models/haarcascade_frontalface_alt2.xml', False)
        # Verifica se o tipo de retorno é um ndarray (ou outro tipo esperado)
        self.assertIsInstance(faces, np.ndarray)

    def test_detect_faces_invalid_haarcascade(self):
        """Testa a detecção de faces com um classificador Haarcascade inválido.

        Verifica se um FileNotFoundError é levantado ao usar um classificador que não existe.
        """
        with self.assertRaises(FileNotFoundError):
            detect_faces(np.zeros((100, 100), dtype=np.uint8), 'invalid_haarcascade.xml')

    def test_detect_landmarks_valid(self):
        """Testa a detecção de marcos faciais em uma imagem que contém faces.

        Verifica se os marcos faciais são detectados corretamente e se possuem a forma esperada.
        """
        image = load_image('test_images/test_face_valid_0.jpg')     
        faces = np.array([[156, 105, 228, 228]])  # Coordenadas dummy de face

        lbf_model = os.path.join("../data/pretrained_models", "lbfmodel.yaml")
        landmark_detector = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel(lbf_model)

        landmarks = detect_landmarks(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), faces, landmark_detector)
        
        # Acesse o primeiro elemento da tupla
        landmarks_array = landmarks[0]  # Acesse o primeiro elemento da tupla
        # Verifica se landmarks_array é um ndarray e se contém marcos
        self.assertIsInstance(landmarks_array, np.ndarray)
        self.assertEqual(landmarks_array.shape[0], 1) 
        self.assertEqual(landmarks_array.shape[1], 68)


    def test_plot_landmarks(self):
        """Testa a plotagem de marcos faciais sobre uma imagem.

        Apenas garante que a função não gera exceções ao ser chamada.
        """        
        image = np.zeros((100, 100, 3), dtype=np.uint8)  # Imagem dummy
        landmarks = [[np.array([[10, 10], [20, 20]])]]  # Marcos dummy
        plot_landmarks(image, landmarks)  # Apenas chama para garantir que não gera exceções

    def test_save_landmarks_to_csv(self):
        """Testa o salvamento de marcos faciais em um arquivo CSV.

        Verifica se os dados estão no formato correto e se uma linha é adicionada.
        """
        landmarks = [np.array([[i, i] for i in range(68)])]  # Marcos dummy com 68 pontos
        save_landmarks_to_csv(landmarks, 1, 0, 'landmarks_test.csv')        
        df = pd.read_csv('landmarks_test.csv')
        
        # Verifica se o número de colunas está correto
        expected_columns = 1 + 1 + 68 + 68  # amostra + class + 68 X + 68 Y
        self.assertEqual(df.shape[1], expected_columns)  # Verifica se o número de colunas está correto
        # Verifica se uma linha foi adicionada
        self.assertEqual(df.shape[0], 1)  # Verifica se 01 linha  foi adicionada

        os.remove('landmarks_test.csv')  # Limpeza após o teste

class TestImageProcessingIntegration(unittest.TestCase):
    """Classe de testes para a integração do processamento de imagens."""

    def test_process_images_in_folder(self):
        """Testa a função process_images_in_folder.

        Garante que as etapas de processamento de imagens estão orquestradas corretamente e que
        os resultados são salvos em um arquivo CSV com os tamanhos corretos para o número de imagens.
        """        
        folder_path = 'test_images'  # Cria uma pasta com imagens de teste
        os.makedirs(folder_path, exist_ok=True)
    
        output_csv = 'test_output/landmarks.csv'
        os.makedirs('test_output', exist_ok=True)
        

        # Chama a função process_images_in_folder
        process_images_in_folder(
            folder_path, 
            '../data/pretrained_models/haarcascade_frontalface_alt2.xml', 
            '../data/pretrained_models/lbfmodel.yaml', 
            output_csv, 
            class_label=0, 
            debug=False
        )
        
        # Verifica se o arquivo CSV foi criado
        self.assertTrue(os.path.exists(output_csv))
        df = pd.read_csv(output_csv)
        self.assertEqual(df.shape[0], 3)  # Verifica se todas as imagens geraram linhas no CSV

        if os.path.isfile(output_csv): 
            os.remove(output_csv)


if __name__ == '__main__':
    unittest.main()
