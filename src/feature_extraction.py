# -*- coding: utf-8 -*-
"""
Módulo de Detecção de Faces e Marcos Faciais
============================================
Este módulo fornece funcionalidades para:
- Carregar uma imagem,
- Detectar faces e marcos faciais,
- Visualizar os resultados,
- Salvar os marcos faciais em um arquivo CSV.

Criado em: Setembro de 2024

@author: George Flores
"""

import cv2
import numpy as np
import pandas as pd
import os
import urllib.request as urlreq
import matplotlib.pyplot as plt


def download_file(url: str, filename: str, debug: bool = False) -> None:
    """
    Faz o download de um arquivo a partir de uma URL e salva localmente.

    Args:
        url (str): URL do arquivo a ser baixado.
        filename (str): Nome do arquivo onde será salvo.
        debug (bool): Se True, exibe informações de debug.

    Returns:
        None
    """
    # Obtém o diretório atual
    current_dir = os.curdir
    full_path = os.path.join(current_dir, filename)

    if not os.path.exists(full_path):
        urlreq.urlretrieve(url, full_path)
        if debug:
            print(f"Arquivo {filename} baixado com sucesso.")
    else:
        if debug:
            print(f"Arquivo {filename} já existe.")


def load_image(image_path: str, debug: bool = False) -> np.ndarray:
    """
    Carrega uma imagem de um caminho especificado.

    Args:
        image_path (str): Caminho do arquivo da imagem.
        debug (bool): Se True, exibe informações de debug.

    Returns:
        np.ndarray: A imagem lida.
    """
    if os.path.exists(image_path):
        if debug:
            print(f"Imagem {image_path} carregada com sucesso.")
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")


def detect_faces(image_gray: np.ndarray, haarcascade: str, debug: bool = False) -> np.ndarray:
    """
    Detecta faces em uma imagem em escala de cinza.

    Args:
        image_gray (np.ndarray): Imagem em escala de cinza.
        haarcascade (str): Caminho do classificador Haarcascade.
        debug (bool): Se True, exibe informações de debug.

    Returns:
        np.ndarray: Coordenadas das faces detectadas.
    """
    if not os.path.exists(haarcascade):
        raise FileNotFoundError(f"Classificador de faces não encontrado: {haarcascade}")

    detector = cv2.CascadeClassifier(haarcascade)
    faces = detector.detectMultiScale(image_gray)

    if debug:
        print(f"Faces detectadas: {faces}")

    return faces


def detect_landmarks(image_gray: np.ndarray, faces: np.ndarray, LBFmodel: str, debug: bool = False) -> list:
    """
    Detecta marcos faciais em uma imagem.

    Args:
        image_gray (np.ndarray): Imagem em escala de cinza.
        faces (np.ndarray): Coordenadas das faces detectadas.
        LBFmodel (str): Caminho do modelo de detecção de marcos faciais.
        debug (bool): Se True, exibe informações de debug.

    Returns:
        list: Lista de marcos faciais detectados.
    """
    if not os.path.exists(LBFmodel):
        raise FileNotFoundError(f"Modelo de marcos faciais não encontrado: {LBFmodel}")

    landmark_detector = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)
    _, landmarks = landmark_detector.fit(image_gray, faces)

    if debug:
        print(f"Marcos faciais detectados: {landmarks}")

    return landmarks


def plot_landmarks(image: np.ndarray, faces: np.ndarray, landmarks: list, debug: bool = False) -> None:
    """
    Plota as faces detectadas e seus marcos faciais na imagem.
    Por questões de privacidade, optou-se por utilizar de imagens brancas.

    Args:
        image (np.ndarray): Imagem original em RGB.
        faces (np.ndarray): Coordenadas das faces detectadas.
        landmarks (list): Lista de marcos faciais detectados.
        debug (bool): Se True, exibe informações de debug.

    Returns:
        None
    """
    # Obtendo dimensões da imagem
    height, width, _ = image.shape
    # Criando uma imagem em branco com as mesmas dimensões da imagem original
    image_blank = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Desenhar os landmarks
    for landmarks_set in landmarks:
        for i, (x, y) in enumerate(landmarks_set[0]):
            cv2.circle(image_blank, (int(x), int(y)), 1, (255, 0, 0), 1)
            cv2.putText(image_blank, str(i + 1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    # Exibindo a imagem resultante com os marcos
    plt.imshow(image_blank)
    plt.axis("off")
    plt.title("Marcos Faciais")
    plt.show()

    if debug:
        print("Landmarks plotados com sucesso.")


def save_landmarks_to_csv(landmarks: list, image_num: int, output_file: str, debug: bool = False) -> None:
    """
    Salva os marcos faciais detectados em um arquivo CSV, acumulando todos os resultados.

    Args:
        landmarks (list): Lista de marcos faciais detectados.
        image_num (int): Número da imagem atual.
        output_file (str): Nome do arquivo CSV onde os marcos serão salvos.
        debug (bool): Se True, exibe informações de debug.

    Returns:
        None
    """
    data = []
    for landmark in landmarks:
        flattened_landmark = [image_num] + landmark.flatten().tolist()
        data.append(flattened_landmark)

    num_points = 68  # Número de pontos faciais padrão
    column_names = ['amostra'] + [f'X{i+1}' for i in range(num_points)] + [f'Y{i+1}' for i in range(num_points)]

    df = pd.DataFrame(data, columns=column_names)

    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
        if debug:
            print(f"Criado arquivo {output_file} e salvou os marcos faciais da imagem {image_num}.")
    else:
        df.to_csv(output_file, mode='a', header=False, index=False)
        if debug:
            print(f"Adicionado marcos faciais da imagem {image_num} ao arquivo {output_file}.")


def process_images_in_folder(folder_path: str, haarcascade: str, LBFmodel: str, output_csv: str, debug: bool = False) -> None:
    """
    Processa todas as imagens em uma pasta, detectando faces e marcos faciais,
    e salvando os resultados em um único CSV.

    Args:
        folder_path (str): Caminho da pasta contendo as imagens.
        haarcascade (str): Caminho do classificador Haarcascade.
        LBFmodel (str): Caminho do modelo de detecção de marcos faciais.
        output_csv (str): Caminho do arquivo CSV onde os marcos faciais serão salvos.
        debug (bool): Se True, exibe informações de debug.

    Returns:
        None
    """
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        if debug:
            print(f"\nProcessando imagem {i + 1}: {image_file}")

        try:
            # Carregar a imagem
            image_rgb = load_image(image_path, debug=debug)
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            # Detectar faces
            faces = detect_faces(image_gray, haarcascade, debug=debug)

            if len(faces) == 0:
                if debug:
                    print(f"Nenhuma face detectada em {image_file}. Pulando para a próxima imagem.")
                continue

            # Detectar marcos faciais
            landmarks = detect_landmarks(image_gray, faces, LBFmodel, debug=debug)

            # Plotar os landmarks
            plot_landmarks(image_rgb, faces, landmarks, debug=debug)

            # Salvar os marcos em um único CSV
            save_landmarks_to_csv(landmarks, i + 1, output_csv, debug=debug)

        except Exception as e:
            if debug:
                print(f"Erro ao processar {image_file}: {e}")


def main():
    # Caminho da pasta com as imagens
    folder_path = '../data/raw/no_autistic'

    # URLs para os arquivos de detecção
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

    # Caminho onde os modelos pré-treinados serão salvos
    models_path = '../data/pretrained_models'
    os.makedirs(models_path, exist_ok=True)

    # Nomes dos arquivos locais
    haarcascade = os.path.join(models_path, "haarcascade_frontalface_alt2.xml")
    LBFmodel = os.path.join(models_path, "LFBmodel.yaml")

    # Baixar os arquivos necessários
    download_file(haarcascade_url, haarcascade, debug=True)
    download_file(LBFmodel_url, LBFmodel, debug=True)

    # Caminho do CSV de saída
    output_csv = 'landmarks_all_images.csv'

    # Processar todas as imagens na pasta
    process_images_in_folder(folder_path, haarcascade, LBFmodel, output_csv, debug=True)


if __name__ == "__main__":
    main()
