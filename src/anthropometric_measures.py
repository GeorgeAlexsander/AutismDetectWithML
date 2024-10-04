# -*- coding: utf-8 -*-
"""
Módulo de Medidas Antropométricas
==================================
Este módulo fornece funcionalidades para:
- Carregar arquivos CSV contendo coordenadas.
- Calcular distâncias antropométricas entre marcos faciais.
- Salvar os resultados em novos arquivos CSV.

Criado em: Sexta-feira, dia 27 de Setembro de 2024
Última modificação em: Sexta-feira, dia 27 de Setembro de 2024

@author: George Flores
"""

import pandas as pd
import numpy as np
import os


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
    Calcula a distância euclidiana entre dois pontos.

    Args:
        point1 (np.ndarray): Primeiro ponto representado como um array NumPy (x, y).
        point2 (np.ndarray): Segundo ponto representado como um array NumPy (x, y).

    Returns:
        float: Distância euclidiana entre os dois pontos.

    Examples:
        >>> point_a = np.array([0, 0])
        >>> point_b = np.array([3, 4])
        >>> calculate_euclidean_distance(point_a, point_b)
        5.0
    """
    return np.linalg.norm(point1 - point2)


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


def calculate_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula as distâncias antropométricas a partir do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contendo as coordenadas dos marcos faciais.

    Returns:
        pd.DataFrame: DataFrame contendo as distâncias calculadas para cada amostra.

    Examples:
        >>> distances_df = calculate_distances(df)
    """
    landmarks = {
        "Trichion": ("X1", "Y1"),
        "Glabella": ("X3", "Y3"),
        "Upper Philtrum": ("X5", "Y5"),
        "Menton": ("X7", "Y7"),
        "Lower Philtrum": ("X9", "Y9"),
        "Endo Canthus Left": ("X11", "Y11"),
        "Endo Canthus Right": ("X13", "Y13"),
        "Exo Canthus Left": ("X15", "Y15"),
        "Exo Canthus Right": ("X17", "Y17"),
        "Alare Left": ("X19", "Y19"),
        "Alare Right": ("X21", "Y21"),
        "Cheilion Left": ("X23", "Y23"),
        "Cheilion Right": ("X25", "Y25"),
    }

    # Inicializando uma lista para armazenar os resultados
    results_list = []

    # Calculando as distâncias para cada amostra
    for index, row in df.iterrows():
        sample = row['amostra']
        class_label = row['class']

        distances = {
            "upper_facial_height": calculate_euclidean_distance(
                np.array([row[landmarks["Trichion"][0]], row[landmarks["Trichion"][1]]]),
                np.array([row[landmarks["Glabella"][0]], row[landmarks["Glabella"][1]]])
            ),
            "middle_facial_height": calculate_euclidean_distance(
                np.array([row[landmarks["Glabella"][0]], row[landmarks["Glabella"][1]]]),
                np.array([row[landmarks["Upper Philtrum"][0]], row[landmarks["Upper Philtrum"][1]]])
            ),
            "lower_facial_height": calculate_euclidean_distance(
                np.array([row[landmarks["Upper Philtrum"][0]], row[landmarks["Upper Philtrum"][1]]]),
                np.array([row[landmarks["Menton"][0]], row[landmarks["Menton"][1]]])
            ),
            "philtrum": calculate_euclidean_distance(
                np.array([row[landmarks["Upper Philtrum"][0]], row[landmarks["Upper Philtrum"][1]]]),
                np.array([row[landmarks["Lower Philtrum"][0]], row[landmarks["Lower Philtrum"][1]]])
            ),
            "intercanthal_width": calculate_euclidean_distance(
                np.array([row[landmarks["Endo Canthus Left"][0]], row[landmarks["Endo Canthus Left"][1]]]),
                np.array([row[landmarks["Endo Canthus Right"][0]], row[landmarks["Endo Canthus Right"][1]]])
            ),
            "biocular_width": calculate_euclidean_distance(
                np.array([row[landmarks["Exo Canthus Left"][0]], row[landmarks["Exo Canthus Left"][1]]]),
                np.array([row[landmarks["Exo Canthus Right"][0]], row[landmarks["Exo Canthus Right"][1]]])
            ),
            "nasal_width": calculate_euclidean_distance(
                np.array([row[landmarks["Alare Left"][0]], row[landmarks["Alare Left"][1]]]),
                np.array([row[landmarks["Alare Right"][0]], row[landmarks["Alare Right"][1]]])
            ),
            "mouth_width": calculate_euclidean_distance(
                np.array([row[landmarks["Cheilion Left"][0]], row[landmarks["Cheilion Left"][1]]]),
                np.array([row[landmarks["Cheilion Right"][0]], row[landmarks["Cheilion Right"][1]]])
            ),
        }

        # Adiciona os resultados à lista
        results_list.append({
            "samples": sample,
            "class": class_label,
            **distances
        })

    # Converte a lista de resultados em um DataFrame
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
    results_no_autism = calculate_distances(df_no_autism)
    results_with_autism = calculate_distances(df_with_autism)

    # Salvar os resultados em novos CSVs
    save_results_to_csv(results_no_autism, output_csv_no_autism)
    save_results_to_csv(results_with_autism, output_csv_with_autism)


if __name__ == "__main__":
    input_csv_no_autism_path = "../data/preprocessed_landmark/landmarks_no_autism.csv"  # caminho do CSV sem autismo
    input_csv_with_autism_path = "../data/preprocessed_landmark/landmarks_with_autism.csv"  # caminho do CSV com autismo
    output_csv_no_autism_path = "../data/preprocessed_landmark/distances_no_autism.csv"  # caminho do CSV de saída sem autismo
    output_csv_with_autism_path = "../data/preprocessed_landmark/distances_with_autism.csv"  # caminho do CSV de saída com autismo
    main(input_csv_no_autism_path, input_csv_with_autism_path, output_csv_no_autism_path, output_csv_with_autism_path)
