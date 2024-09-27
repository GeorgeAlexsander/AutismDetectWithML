# -*- coding: utf-8 -*-
"""
Módulo de Extração de Features
==============================
Este módulo fornece funcionalidades para download de imagens a partir de URLs,
processamento utilizando OpenCV e visualização usando Matplotlib.

Criado em: Quarta-feira, 11 de Setembro de 2024
Alterado por último em: Quinta-feira, 26 de Setembro de 2024

@author: George Flores

Funções
--------
- download_image_from_url(url: str, filename: str) -> None: Faz o download de uma imagem a partir de uma URL.
- read_image(filename: str) -> np.ndarray: Lê uma imagem de um arquivo.
- process_image(image: np.ndarray) -> np.ndarray: Aplica processamento básico em uma imagem.
- display_image(image: np.ndarray, title: str = "Imagem") -> None: Exibe uma imagem usando Matplotlib.
- save_processed_image(image: np.ndarray, output_filename: str) -> None: Salva uma imagem processada.
"""

import cv2
import numpy as np
import urllib.request as urlreq
import matplotlib.pyplot as plt
from pylab import rcParams

print(cv2.__version__)

# Define o tamanho das imagens para exibição
rcParams['figure.figsize'] = 10, 10

def download_image_from_url(url: str, filename: str) -> None:
    """
    Faz o download de uma imagem a partir de uma URL e a salva com o nome de arquivo especificado.

    Parameters
    ----------
    url : str
        A URL da imagem a ser baixada.
    filename : str
        O nome do arquivo onde a imagem será salva.

    Returns
    -------
    None
    """
    try:
        urlreq.urlretrieve(url, filename)
        print(f"Imagem baixada e salva como {filename}")
    except Exception as e:
        print(f"Erro ao baixar a imagem: {e}")

def read_image(filename: str) -> np.ndarray:
    """
    Lê uma imagem do arquivo especificado e a converte para o formato BGR.

    Parameters
    ----------
    filename : str
        O nome do arquivo da imagem a ser lida.

    Returns
    -------
    image : np.ndarray
        A imagem em formato BGR.
    """
    try:
        image = cv2.imread(filename)
        if image is None:
            raise FileNotFoundError(f"Arquivo {filename} não encontrado.")
        return image
    except Exception as e:
        print(f"Erro ao ler a imagem: {e}")
        return None

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Aplica processamento básico na imagem, convertendo-a para escala de cinza e redimensionando-a.

    Parameters
    ----------
    image : np.ndarray
        A imagem original em formato BGR.

    Returns
    -------
    processed_image : np.ndarray
        A imagem processada em escala de cinza e redimensionada.
    """
    try:
        # Conversão para escala de cinza
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Redimensionamento da imagem para um tamanho fixo
        resized_image = cv2.resize(gray_image, (100, 100))
        return resized_image
    except Exception as e:
        print(f"Erro no processamento da imagem: {e}")
        return None

def display_image(image: np.ndarray, title: str = "Imagem") -> None:
    """
    Exibe uma imagem usando Matplotlib.

    Parameters
    ----------
    image : np.ndarray
        A imagem a ser exibida.
    title : str, optional
        O título da janela de exibição (padrão é "Imagem").

    Returns
    -------
    None
    """
    try:
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Erro ao exibir a imagem: {e}")

def save_processed_image(image: np.ndarray, output_filename: str) -> None:
    """
    Salva uma imagem processada em um arquivo especificado.

    Parameters
    ----------
    image : np.ndarray
        A imagem processada a ser salva.
    output_filename : str
        O nome do arquivo onde a imagem será salva.

    Returns
    -------
    None
    """
    try:
        cv2.imwrite(output_filename, image)
        print(f"Imagem processada salva como {output_filename}")
    except Exception as e:
        print(f"Erro ao salvar a imagem processada: {e}")
