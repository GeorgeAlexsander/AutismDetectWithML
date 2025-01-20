# -*- coding: utf-8 -*-
"""
FaceMesh3DExtractor
====================
Este módulo utiliza o MediaPipe para:
- Carregar uma imagem,
- Detectar faces e marcos faciais em 3D,
- Visualizar e salvar os resultados em um arquivo CSV.

Criado em: Sexta-feira, dia 27 de Setembro de 2024
Última modificação em: Sexta-feira, dia 27 de Setembro de 2024

@author: George Flores
"""

import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Inicializa a solução Face Mesh do MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Caminhos das pastas de entrada e saída
INPUT_FOLDER_NO_AUTISM = "../data/data_processing/data_M-C-A-F/valid/Non_Autistic"
INPUT_FOLDER_WITH_AUTISM = "../data/data_processing/data_M-C-A-F/valid/Autistic"
OUTPUT_FOLDER = "../data/preprocessed_landmark/face_mesh/data_processing/data_M-C-A-F"

def load_image(image_path: str, debug: bool = False) -> np.ndarray:
    """
    Carrega uma imagem de um caminho especificado.

    Args:
        image_path (str): Caminho do arquivo da imagem.
        debug (bool): Se True, exibe informações de debug.

    Returns:
        np.ndarray: A imagem lida em formato RGB.

    Raises:
        FileNotFoundError: Se o arquivo da imagem não for encontrado.
    """
    if os.path.exists(image_path):
        if debug:
            print(f"Imagem {image_path} carregada com sucesso.")
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")


def detect_face_mesh(image_rgb: np.ndarray, debug: bool = False) -> list:
    """
    Detecta marcos faciais 3D usando o MediaPipe FaceMesh.

    Args:
        image_rgb (np.ndarray): Imagem RGB carregada.
        debug (bool): Se True, exibe informações de debug.

    Returns:
        list: Lista de marcos faciais 3D, onde cada conjunto contém as coordenadas (x, y, z).
    """
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(image_rgb)
        landmarks_3d = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for lm in face_landmarks.landmark:
                    # Converte as coordenadas de normalizadas para pixel
                    height, width, _ = image_rgb.shape
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    z = lm.z  # Z permanece em valor normalizado
                    landmarks_3d.append((x, y, z))

        if debug and len(landmarks_3d) > 0:
            print(f"{len(landmarks_3d)} marcos faciais detectados em 3D.")
        
        return landmarks_3d


def plot_landmarks(image: np.ndarray, landmarks: list, debug: bool = False) -> None:
    """
    Plota os marcos faciais detectados na imagem.

    Args:
        image (np.ndarray): Imagem original em RGB.
        landmarks (list): Lista de marcos faciais detectados.
        debug (bool): Se True, exibe informações de debug.

    Returns:
        None
    """
    # Desenhar os landmarks
    for (x, y, z) in landmarks:
        cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

    # Exibindo a imagem resultante com os marcos
    if debug:
        plt.imshow(image)
        plt.axis("off")
        plt.title("Marcos Faciais 3D")
        plt.show()


def plot_main_landmarks(image: np.ndarray, landmarks: list, debug: bool = False) -> None:
    """
    Plota os principais marcos faciais na imagem em vermelho.

    Args:
        image (np.ndarray): Imagem original em RGB.
        landmarks (list): Lista de marcos faciais detectados.
        debug (bool): Se True, exibe informações de debug.

    Returns:
        None
    """
    # Definindo os índices dos principais marcos faciais
    main_landmark_indices = {
        "Trichion": 10,
        "Glabella": 9,
        "Endo Canthus Left": 133,
        "Endo Canthus Right": 362,
        "Exo Canthus Left": 263,
        "Exo Canthus Right": 33,
        "Upper Philtrum": 19,
        "Alare Left": 294,
        "Alare Right": 64,
        "Lower Philtrum": 0,
        "Christa Philtri Left": 267,
        "Christa Philtri Right": 37,
        "Cheilion Left": 61,
        "Cheilion Right": 291,
        "Pogonion": 199,
        "Menton": 152,
        "frontozygomaticus_left": 300,
        "frontozygomaticus_right": 70
    }

    # Criar uma imagem branca com as mesmas dimensões da original
    blank_image = np.ones_like(image) * 255  # Imagem branca (255 para cada canal RGB)

    # Desenhar os principais marcos na imagem
    for name, index in main_landmark_indices.items():
        if index < len(landmarks):  # Verifica se o índice está dentro do alcance
            x, y, _ = landmarks[index]
            # Desenha um círculo vermelho para o marco
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)  # Círculo vermelho
            # Adiciona o número do marco ao lado
            cv2.putText(image, str(index), (int(x) + 2, int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)

    # Exibindo a imagem resultante com os marcos principais
    if debug:
        plt.imshow(image)
        plt.axis("off")
        plt.title("Marcos Faciais Principais")
        plt.show()


def save_landmarks_to_csv(landmarks: list, image_num: int, class_label: int, output_file: str, debug: bool = False) -> None:
    """
    Salva os marcos faciais detectados em um arquivo CSV, organizados como X1, Y1, Z1, X2, Y2, Z2, etc.

    Args:
        landmarks (list): Lista de marcos faciais detectados.
        image_num (int): Número da imagem atual.
        output_file (str): Nome do arquivo CSV onde os marcos serão salvos.
        debug (bool): Se True, exibe informações de debug.

    Returns:
        None
    """
    # Inicializa a lista de dados
    flattened_landmarks = [image_num, class_label]

    # Adiciona as coordenadas X, Y, Z de cada ponto facial
    for x, y, z in landmarks:
        flattened_landmarks.extend([x, y, z])

    # Define os nomes das colunas: X1, Y1, Z1, X2, Y2, Z2, ..., X468, Y468, Z468
    columns = ["amostra", "class"]
    for i in range(len(landmarks)):
        columns.extend([f"X{i}", f"Y{i}", f"Z{i}"])

    # Verifica se o arquivo já existe e salva os dados
    df = pd.DataFrame([flattened_landmarks], columns=columns)
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
        if debug:
            print(f"Criado arquivo {output_file} e salvou os marcos faciais da imagem {image_num}.")
    else:
        df.to_csv(output_file, mode="a", header=False, index=False)
        if debug:
            print(f"Adicionado marcos faciais da imagem {image_num} ao arquivo {output_file}.")


def process_images_in_folder(
    folder_path: str, output_csv: str, class_label: int, debug: bool = False
) -> None:
    """
    Processa todas as imagens em uma pasta, detectando marcos faciais 3D,
    e salvando os resultados em um único CSV.

    Args:
        folder_path (str): Caminho da pasta contendo as imagens.
        output_csv (str): Caminho do arquivo CSV onde os marcos faciais serão salvos.
        class_label (int): Rótulo da classe para a imagem (0 para sem autismo, 1 para com autismo).
        debug (bool): Se True, exibe informações de debug.

    Returns:
        None
    """
    image_files = [
        f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    for i, image_file in enumerate(tqdm(image_files)):
        image_path = os.path.join(folder_path, image_file)
        try:
            image_rgb = load_image(image_path, debug)
            landmarks_3d = detect_face_mesh(image_rgb, debug)

            if landmarks_3d:
                save_landmarks_to_csv(landmarks_3d, i + 1, class_label, output_csv, debug)

                # Plotar a imagem com os marcos (opcional)
                plot_landmarks(image_rgb.copy(), landmarks_3d, debug)
                plot_main_landmarks(image_rgb.copy(), landmarks_3d, debug)
        except FileNotFoundError as e:
            if debug:
                print(e)


def main():
    """
    Função principal para processar as imagens e gerar os arquivos CSV de marcos faciais.
    """
    # Caminhos das pastas de entrada
    no_autism_folder = INPUT_FOLDER_NO_AUTISM
    with_autism_folder = INPUT_FOLDER_WITH_AUTISM

    # Caminho de saída para os arquivos CSV
    output_no_autism = os.path.join(OUTPUT_FOLDER, "face_mesh_M-C-A-F_valid_no_autism.csv")
    output_with_autism = os.path.join(OUTPUT_FOLDER, "face_mesh_M-C-A-F_valid_autism.csv")

    # Processa as imagens sem autismo
    process_images_in_folder(no_autism_folder, output_no_autism, class_label=0, debug=False)

    # Processa as imagens com autismo
    process_images_in_folder(with_autism_folder, output_with_autism, class_label=1, debug=False)


if __name__ == "__main__":
    main()
