# -*- coding: utf-8 -*-
"""
Módulo de Medidas Antropométricas com Face Mesh 3D
=====================================================
Este módulo fornece funcionalidades para:
- Carregar arquivos CSV contendo coordenadas 3D dos marcos faciais.
- Calcular distâncias antropométricas entre marcos faciais.
- Salvar os resultados em novos arquivos CSV.

Criado em: Sexta-feira, dia 27 de Setembro de 2024
Última modificação em: Sexta-feira, dia 27 de Setembro de 2024

@author: George Flores
"""

import pandas as pd
import numpy as np
import os
from scipy.spatial import distance

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Carrega um arquivo CSV em um DataFrame.

    Args:
        file_path (str): Caminho do arquivo CSV a ser carregado.

    Returns:
        pd.DataFrame: DataFrame contendo os dados do CSV.

    Raises:
        FileNotFoundError: Se o arquivo CSV não for encontrado.

    Examples:
        >>> df = load_csv("caminho/do/arquivo.csv")
    """
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

def calculate_euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calcula a distância euclidiana entre dois pontos 2D (X, Y).

    Args:
        point1 (np.ndarray): Primeiro ponto representado como um array NumPy (x, y).
        point2 (np.ndarray): Segundo ponto representado como um array NumPy (x, y).

    Returns:
        float: Distância euclidiana entre os dois pontos.

    Examples:
        >>> point_a = np.array([0, 0])
        >>> point_b = np.array([3, 4])
        >>> calculate_euclidean_distance_2d(point_a, point_b)
        5.0
    """
    # Verifica se qualquer ponto é inválido (-1, -1)
    if np.array_equal(point1, [-1, -1]) or np.array_equal(point2, [-1, -1]):
        return -1  # Indica que a distância é inválida
    return distance.euclidean(point1, point2)

def save_results_to_csv(results: pd.DataFrame, output_file: str) -> None:
    """
    Salva os resultados em um arquivo CSV.

    Args:
        results (pd.DataFrame): DataFrame contendo os resultados a serem salvos.
        output_file (str): Caminho do arquivo CSV onde os resultados serão salvos.

    Returns:
        None

    Examples:
        >>> save_results_to_csv(results_df, "caminho/do/resultado.csv")
    """
    results.to_csv(output_file, index=False)

def calculate_distances_3d(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Calcula as distâncias antropométricas em 3D a partir do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contendo as coordenadas dos marcos faciais em 3D.
        debug (bool): Se True, imprime os arrays de pontos usados para calcular as distâncias.

    Returns:
        pd.DataFrame: DataFrame contendo as distâncias calculadas para cada amostra.

    Examples:
        >>> distances_df = calculate_distances_3d(df, debug=True)
    """
    results_list = []

    for index, row in df.iterrows():
        sample = row['amostra']
        class_label = row['class']

        # Definindo alguns dos marcos faciais principais
        # Tamanhos da face
        face_width_ref1 = np.array([row['X127'], row['Y127']])
        face_width_ref2 = np.array([row['X356'], row['Y356']])
        # Parte Superior do rosto
        trichion = np.array([row['X10'], row['Y10']])    
        glabella = np.array([row['X9'], row['Y9']])
        frontozygomaticus_left = np.array([row['X300'], row['Y300']])
        frontozygomaticus_right = np.array([row['X70'], row['Y70']])
        # Olhos
        endo_canthus_left = np.array([row['X133'], row['Y133']])
        endo_canthus_right = np.array([row['X362'], row['Y362']])
        exo_canthus_left = np.array([row['X263'], row['Y263']])
        exo_canthus_right = np.array([row['X33'], row['Y33']])
        # Nariz
        upper_philtrum = np.array([row['X19'], row['Y19']])
        alare_left = np.array([row['X294'], row['Y294']])
        alare_right = np.array([row['X64'], row['Y64']])
        # Labios
        lower_philtrum = np.array([row['X0'], row['Y0']])
        christa_philtri_left = np.array([row['X267'], row['Y267']])
        christa_philtri_right = np.array([row['X37'], row['Y37']])
        cheilion_left = np.array([row['X61'], row['Y61']])
        cheilion_right = np.array([row['X291'], row['Y291']])
        # Queixo
        pogonion = np.array([row['X199'], row['Y199']])
        menton = np.array([row['X152'], row['Y152']])
        distances = {
        
            # Distâncias básicas
            "upper_facial_height": calculate_euclidean_distance(trichion, glabella),
            "middle_facial_height": calculate_euclidean_distance(glabella, menton),
            "intercanthal_width": calculate_euclidean_distance(endo_canthus_left, endo_canthus_right),
            "biocular_width": calculate_euclidean_distance(exo_canthus_left, exo_canthus_right),
            "nasal_width": calculate_euclidean_distance(alare_left, alare_right),
            "mouth_width": calculate_euclidean_distance(cheilion_left, cheilion_right),
            "philtrum_height": calculate_euclidean_distance(upper_philtrum, lower_philtrum),
            # Distancias sugeridas por artigo da literatura, medições padronizadas de cima para baixo, da esquerda p/ direita
            "eye_left_width": calculate_euclidean_distance(exo_canthus_left, endo_canthus_left),
            "eye_right_width": calculate_euclidean_distance(endo_canthus_right, exo_canthus_right),
            "endo_canthus_glabella_left": calculate_euclidean_distance(endo_canthus_left, glabella),
            "endo_canthus_glabella_right": calculate_euclidean_distance(glabella, endo_canthus_right),
            "exo_canthus_christa_philtri_left": calculate_euclidean_distance(christa_philtri_left, exo_canthus_left),
            "exo_canthus_christa_philtri_right": calculate_euclidean_distance(exo_canthus_right, christa_philtri_right),
            "alare_left_lower_philtrum": calculate_euclidean_distance(alare_left,lower_philtrum),
            "glabella_alare_right": calculate_euclidean_distance(glabella, alare_right),
            "glabella_christa_philtri_left": calculate_euclidean_distance(glabella, christa_philtri_left),
            "glabella_lower_philtrum": calculate_euclidean_distance(glabella, lower_philtrum),
            "glabella_christa_philtri_right": calculate_euclidean_distance(glabella, christa_philtri_right),            
            "christa_philtri_right_alare_left": calculate_euclidean_distance(alare_left, christa_philtri_right),
            "christa_philtri_right_cheilion_left": calculate_euclidean_distance(cheilion_left, christa_philtri_right),
            "christa_philtri_left_cheilion_right": calculate_euclidean_distance(christa_philtri_left, cheilion_right),
            "christa_philtri_left_cheilion_right": calculate_euclidean_distance(christa_philtri_left, cheilion_right),
            "christa_philtri_left_cheilion_right": calculate_euclidean_distance(christa_philtri_left, cheilion_right),
            "christa_philtri_left_cheilion_right": calculate_euclidean_distance(christa_philtri_left, cheilion_right),
            "christa_philtri_left_lower_philtrum": calculate_euclidean_distance(lower_philtrum,christa_philtri_left),
            "cheilion_left_lower_philtrum": calculate_euclidean_distance(cheilion_left, lower_philtrum),
            "cheilion_left_christa_philtri_right": calculate_euclidean_distance(cheilion_left, christa_philtri_right),
            "cheilion_left_christa_philtri_left": calculate_euclidean_distance(cheilion_left, christa_philtri_left),
            "cheilion_left_cheilion_right": calculate_euclidean_distance(cheilion_left, cheilion_right),
            "cheilion_left_pogonion": calculate_euclidean_distance(cheilion_left, pogonion),
            "cheilion_right_lower_philtrum": calculate_euclidean_distance(cheilion_right, lower_philtrum),
            "cheilion_right_christa_philtri_right": calculate_euclidean_distance(cheilion_right, christa_philtri_right),
            "cheilion_right_christa_philtri_left": calculate_euclidean_distance(cheilion_right, christa_philtri_left),
            "frontozygomaticus_endo_cantus_left": calculate_euclidean_distance(frontozygomaticus_left, exo_canthus_left),
            "frontozygomaticus_exo_cantus_left": calculate_euclidean_distance(frontozygomaticus_left, exo_canthus_left),
            "frontozygomaticus_left_alare_right": calculate_euclidean_distance(frontozygomaticus_left, alare_right),
            "frontozygomaticus_left_cheilion_right": calculate_euclidean_distance(frontozygomaticus_left, cheilion_right),
            "frontozygomaticus_endo_cantus_right": calculate_euclidean_distance(frontozygomaticus_right, endo_canthus_right),
            "frontozygomaticus_exo_cantus_right": calculate_euclidean_distance(frontozygomaticus_right, exo_canthus_right),
            "frontozygomaticus_right_cheilion_left": calculate_euclidean_distance(frontozygomaticus_right, cheilion_left),
            "face_height": calculate_euclidean_distance(trichion, menton),
            "face_width": calculate_euclidean_distance(face_width_ref2, face_width_ref1)
        }

        if debug and index == 0:  # Exibir apenas para a primeira amostra se o debug estiver ativado
            print(f"Calculando distâncias para a amostra: {sample}")
            for key, value in distances.items():
                print(f"{key}: {value}")

        results_list.append({
            "samples": sample,
            "class": class_label,
            **distances
        })

    return pd.DataFrame(results_list)

def main(input_csv_no_autism: str, input_csv_with_autism: str, output_csv_no_autism: str, output_csv_with_autism: str) -> None:
    """
    Função principal que orquestra o cálculo das distâncias para ambos os grupos.

    Args:
        input_csv_no_autism (str): Caminho do arquivo CSV contendo dados sem autismo.
        input_csv_with_autism (str): Caminho do arquivo CSV contendo dados com autismo.
        output_csv_no_autism (str): Caminho do arquivo CSV onde os resultados sem autismo serão salvos.
        output_csv_with_autism (str): Caminho do arquivo CSV onde os resultados com autismo serão salvos.

    Returns:
        None

    Examples:
        >>> main("path/to/no_autism.csv", "path/to/with_autism.csv", "output/no_autism_distances.csv", "output/with_autism_distances.csv")
    """
    # Carregar os dados
    df_no_autism = load_csv(input_csv_no_autism)
    df_with_autism = load_csv(input_csv_with_autism)

    # Calcular as distâncias para cada grupo
    results_no_autism = calculate_distances_3d(df_no_autism)
    results_with_autism = calculate_distances_3d(df_with_autism)

    # Salvar os resultados em novos CSVs
    save_results_to_csv(results_no_autism, output_csv_no_autism)
    save_results_to_csv(results_with_autism, output_csv_with_autism)

if __name__ == "__main__":
    input_csv_no_autism_path = "../data/preprocessed_landmark/face_mesh_no_autism_6.0.csv"  # caminho do CSV sem autismo
    input_csv_with_autism_path = "../data/preprocessed_landmark/face_mesh_with_autism_6.0.csv"  # caminho do CSV com autismo
    output_csv_no_autism_path = "../data/preprocessed_landmark/face_mesh_distances_no_autism_6.0.csv"  # caminho do CSV de saída sem autismo
    output_csv_with_autism_path = "../data/preprocessed_landmark/face_mesh_distances_with_autism_6.0.csv"  # caminho do CSV de saída com autismo
    main(input_csv_no_autism_path, input_csv_with_autism_path, output_csv_no_autism_path, output_csv_with_autism_path)