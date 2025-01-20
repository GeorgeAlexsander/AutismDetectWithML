import cv2
import mediapipe as mp
import numpy as np
import os
import math

# Inicializando MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


def calculate_angle(vector1, vector2):
    """
    Calcula o ângulo entre dois vetores.

    Args:
        vector1 (list): Coordenadas do primeiro vetor [x, y].
        vector2 (list): Coordenadas do segundo vetor [x, y].

    Returns:
        float: Ângulo entre os vetores em graus.
    """
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a**2 for a in vector1))
    magnitude2 = math.sqrt(sum(b**2 for b in vector2))
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    angle = math.degrees(math.acos(cosine_similarity))
    return angle


def calculate_face_orientation(landmarks):
    """
    Calcula a orientação da face com base nos landmarks faciais.

    Args:
        landmarks (list): Lista de landmarks faciais de MediaPipe.

    Returns:
        float: Ângulo de orientação facial em graus.
    """
    vector1 = [landmarks[9].x - landmarks[19].x, landmarks[9].y - landmarks[19].y]  # Sobrancelha ao nariz
    vector2 = [landmarks[64].x - landmarks[294].x, landmarks[64].y - landmarks[294].y]  # Narinas
    angle = calculate_angle(vector1, vector2)
    return angle


def normalize_image(image):
    """
    Normaliza uma imagem para valores entre 0 e 1.

    Args:
        image (np.ndarray): Imagem em formato numpy array.

    Returns:
        np.ndarray: Imagem normalizada.
    """
    normalized_image = image.astype(np.float32) / 255.0
    return normalized_image


def process_images(input_directory, output_directory, orientation_range=(80, 100)):
    """
    Processa imagens em um diretório, filtrando por orientação frontal e normalizando.

    Args:
        input_directory (str): Caminho para o diretório de entrada com imagens.
        output_directory (str): Caminho para salvar imagens normalizadas.
        orientation_range (tuple): Intervalo de ângulos aceitos para orientação frontal.

    Returns:
        list: Lista de nomes de arquivos que atendem aos critérios de orientação.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    filtered_images = []

    for img_file in os.listdir(input_directory):
        img_path = os.path.join(input_directory, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                face_orientation = calculate_face_orientation(landmarks)

                if orientation_range[0] <= face_orientation <= orientation_range[1]:
                    filtered_images.append(img_file)

                    normalized_image = normalize_image(image)
                    output_path = os.path.join(output_directory, img_file)
                    cv2.imwrite(output_path, (normalized_image * 255).astype(np.uint8))

    return filtered_images


def main():
    """
    Função principal que processa imagens para ambos os tipos (autistic e non-autistic).
    """
    # Configuração de diretórios
    base_directory = "../data/cleanData/"  # Atualize para o caminho correto
    input_dirs = {
        "autistic": os.path.join(base_directory, "with_autistic"),
        "non_autistic": os.path.join(base_directory, "no_Autistic")
    }
    output_dirs = {
        "autistic": os.path.join(base_directory, "with_autistic_normalized"),
        "non_autistic": os.path.join(base_directory, "no_autistic_normalized")
    }

    # Processar ambos os diretórios
    for label, input_dir in input_dirs.items():
        output_dir = output_dirs[label]
        print(f"Processando imagens para {label}...")
        filtered_images = process_images(input_dir, output_dir)
        print(f"{len(filtered_images)} imagens filtradas e normalizadas salvas em {output_dir}.")


if __name__ == "__main__":
    main()
