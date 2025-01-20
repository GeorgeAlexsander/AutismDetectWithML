import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

# Índices dos principais landmarks necessários
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

def align_image(image, face_landmarks, image_width, image_height, debug=False):
    """
    Alinha a imagem com base nos pontos dos olhos.

    Args:
        image (numpy.ndarray): Imagem original.
        face_landmarks: Objeto de landmarks detectados pelo FaceMesh.
        image_width (int): Largura da imagem.
        image_height (int): Altura da imagem.
        debug (bool): Se True, exibe mensagens de debug.

    Returns:
        numpy.ndarray: Imagem alinhada ou a original em caso de falha.
    """
    try:
        # Coordenadas normalizadas dos pontos dos olhos
        right_eye = face_landmarks.landmark[33]
        left_eye = face_landmarks.landmark[263]

        # Converte coordenadas normalizadas para coordenadas de pixels
        right_eye_pixel = _normalized_to_pixel_coordinates(
            right_eye.x, right_eye.y, image_width, image_height
        )
        left_eye_pixel = _normalized_to_pixel_coordinates(
            left_eye.x, left_eye.y, image_width, image_height
        )

        # Verifica se os pontos dos olhos foram detectados
        if right_eye_pixel is None or left_eye_pixel is None:
            if debug:
                print("Erro: Coordenadas dos olhos não detectadas corretamente.")
            return image

        # Extrai as coordenadas dos olhos
        right_eye_x, right_eye_y = right_eye_pixel
        left_eye_x, left_eye_y = left_eye_pixel

        # Calcula o ângulo de rotação
        delta_x = left_eye_x - right_eye_x
        delta_y = left_eye_y - right_eye_y
        angle = np.arctan2(delta_y, delta_x) * (180.0 / np.pi)

        if debug:
            print(f"Ângulo de rotação calculado: {angle:.2f} graus")

        # Calcula o centro da rotação como o ponto médio entre os olhos
        center = ((right_eye_x + left_eye_x) // 2, (right_eye_y + left_eye_y) // 2)

        # Gera a matriz de rotação
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

        # Aplica a rotação à imagem
        aligned_image = cv2.warpAffine(image, rotation_matrix, (image_width, image_height))

        return aligned_image

    except Exception as e:
        if debug:
            print(f"Erro ao alinhar imagem: {e}")
        return image


def crop_face(image, face_landmarks, image_width, image_height, padding=20):
    """
    Realiza o recorte da área da face detectada na imagem.

    Args:
        image (numpy.ndarray): Imagem original.
        face_landmarks: Objeto de landmarks detectados pelo FaceMesh.
        image_width (int): Largura da imagem.
        image_height (int): Altura da imagem.
        padding (int): Margem extra ao redor da área da face.

    Returns:
        numpy.ndarray: Imagem recortada na área da face.
    """
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0

    for lm in face_landmarks.landmark:
        pixel_coordinates = _normalized_to_pixel_coordinates(
            lm.x, lm.y, image_width, image_height
        )
        if pixel_coordinates:
            x, y = pixel_coordinates
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

    # Adiciona padding e garante que o recorte esteja dentro dos limites da imagem
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_width, x_max + padding)
    y_max = min(image_height, y_max + padding)

    return image[y_min:y_max, x_min:x_max]

def process_images_in_folder(
    folder_path: str, output_folder: str, debug: bool = False
) -> tuple:
    """
    Processa todas as imagens em uma pasta usando FaceMesh.

    Args:
        folder_path (str): Caminho da pasta contendo as imagens.
        output_folder (str): Caminho da pasta onde as imagens serão salvas.
        debug (bool): Se True, exibe mensagens de debug e erros.

    Returns:
        tuple: Quantidade de imagens que passaram e que não passaram no teste.
    """
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1
    )

    # Cria a pasta de saída, se não existir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_passed = 0
    total_failed = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not (filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
            if debug:
                print(f"Arquivo ignorado: {filename}")
            continue

        image = cv2.imread(file_path)
        if image is None:
            if debug:
                print(f"Erro ao carregar imagem: {file_path}")
            continue

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    image_height, image_width, _ = image.shape

                    aligned_image = align_image(image, face_landmarks, image_width, image_height, debug)

                    cropped_face = crop_face(aligned_image, face_landmarks, image_width, image_height)
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, cropped_face)
                    total_passed += 1
                    if debug:
                        print(f"Imagem salva: {output_path}")
                    break
            else:
                total_failed += 1
                if debug:
                    print(f"Nenhuma face detectada: {filename}")

        except Exception as e:
            total_failed += 1
            if debug:
                print(f"Erro ao processar {filename}: {e}")

    return total_passed, total_failed


if __name__ == "__main__":
    # Define os diretórios de entrada e saída
    input_folders = {
        "Autistic": "../data/raw_sem_duplicatas_exclusao_manual/Autistic",
        "Non_Autistic": "../data/raw_sem_duplicatas_exclusao_manual/Non_Autistic"
    }
    output_base_folder = "../data/data_exclusao_manual"

    # Contadores globais
    global_passed = 0
    global_failed = 0

    # Processa as imagens para cada diretório
    for label, folder in input_folders.items():
        output_folder = os.path.join(output_base_folder, label)
        print(f"Processando imagens da pasta: {folder}")
        passed, failed = process_images_in_folder(folder, output_folder, debug=True)
        global_passed += passed
        global_failed += failed

        print(f"Resultados para a pasta '{label}':")
        print(f"  Imagens que passaram: {passed}")
        print(f"  Imagens que falharam: {failed}")

    # Exibe o total geral
    print("\nProcessamento concluído!")
    print(f"Total de imagens que passaram: {global_passed}")
    print(f"Total de imagens que falharam: {global_failed}")
