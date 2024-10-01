# src/__init__.py

"""
Pacote de Detecção de Faces e Marcos Faciais.

Este módulo fornece funcionalidades para detectar faces e marcos faciais em imagens,
visualizar os resultados e salvar os marcos em um arquivo CSV.
"""

# Importando as funções principais para facilitar o acesso
from .feature_extraction import (
    download_file,
    load_image,
    detect_faces,
    detect_landmarks,
    plot_landmarks,
    save_landmarks_to_csv,
    process_images_in_folder
)
