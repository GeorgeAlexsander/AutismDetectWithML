import cv2
import mediapipe as mp
import numpy as np
import os
import shutil

# Inicializar o MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def initialize_face_mesh():
    """
    Inicializa a configuração do MediaPipe FaceMesh.

    Returns:
        FaceMesh: Instância do modelo FaceMesh configurado.
    """
    return mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True
    )

def process_image(image_path, face_mesh):
    """
    Processa a imagem para detectar landmarks faciais e calcular os ângulos de orientação.

    Args:
        image_path (str): Caminho da imagem a ser processada.
        face_mesh: Instância do modelo FaceMesh configurado.

    Returns:
        tuple: Roll, Pitch, Yaw em graus, ou None se não houver landmarks detectados.
    """
    # Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Converter para RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Processar a imagem
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    for landmarks in results.multi_face_landmarks:
        # Selecionar landmarks específicas para cálculos
        points_2d = extract_landmarks(landmarks, image)

        # Calcular ângulos de orientação (roll, pitch, yaw)
        roll, pitch, yaw = calculate_angles(points_2d)

        return roll, pitch, yaw

    return None

def extract_landmarks(landmarks, image):
    """
    Extrai os pontos de interesse da face para cálculo.

    Args:
        landmarks: Landmarks faciais detectados.
        image: Imagem original para uso nas dimensões de coordenadas.

    Returns:
        dict: Coordenadas 2D dos pontos de interesse (ex: ponta do nariz, olhos, boca, etc).
    """
    def to_2d(landmark):
        return int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])

    # Coordenadas de pontos específicos da face
    return {
        "nose_tip": to_2d(landmarks.landmark[1]),
        "left_eye": to_2d(landmarks.landmark[33]),
        "right_eye": to_2d(landmarks.landmark[263]),
        "mouth_left": to_2d(landmarks.landmark[61]),
        "mouth_right": to_2d(landmarks.landmark[291]),
        "chin": to_2d(landmarks.landmark[199]),
    }

def calculate_angles(points_2d):
    """
    Calcula os ângulos de orientação (roll, pitch, yaw) a partir dos pontos 2D.

    Args:
        points_2d (dict): Coordenadas 2D dos pontos de interesse.

    Returns:
        tuple: Roll, Pitch, Yaw em graus.
    """
    # Calcular os vetores entre os pontos
    left_to_right = np.array(points_2d["right_eye"]) - np.array(points_2d["left_eye"])
    nose_to_chin = np.array(points_2d["chin"]) - np.array(points_2d["nose_tip"])

    # Normalizar os vetores
    left_to_right = left_to_right / np.linalg.norm(left_to_right)
    nose_to_chin = nose_to_chin / np.linalg.norm(nose_to_chin)

    # Estimar os ângulos (em radianos)
    roll = np.arctan2(left_to_right[1], left_to_right[0])  # Inclinação lateral
    pitch = np.arctan2(nose_to_chin[0], nose_to_chin[1])   # Inclinação frontal
    yaw = np.arctan2(left_to_right[1], nose_to_chin[1])    # Rotação em torno do eixo vertical

    # Converter para graus
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def create_output_folders(output_base_folder):
    """
    Cria as pastas de saída necessárias.

    Args:
        output_base_folder (str): Caminho da pasta base onde as imagens serão salvas.
    """
    for folder in ["Autistic", "Non_Autistic", "Remove_Autistic", "Remove_Non_Autistic"]:
        folder_path = os.path.join(output_base_folder, folder)
        os.makedirs(folder_path, exist_ok=True)

def copy_image_to_folder(image_path, output_folder):
    """
    Copia uma imagem para a pasta de saída correspondente.

    Args:
        image_path (str): Caminho da imagem original.
        output_folder (str): Caminho da pasta de destino.
    """
    shutil.copy(image_path, output_folder)

def process_images_in_folder(input_folders, output_base_folder, angle_threshold=7, debug=False):
    """
    Processa todas as imagens nas pastas de entrada usando FaceMesh e organiza as imagens em pastas com base nos ângulos estimados.

    Args:
        input_folders (dict): Dicionário com os caminhos das pastas de entrada para imagens autistas e não autistas.
        output_base_folder (str): Caminho da pasta onde as imagens processadas serão salvas.
        angle_threshold (float): Limite para os ângulos roll, pitch e yaw. Imagens com ângulos abaixo desse valor serão copiadas para a pasta correspondente. Padrão é 20 graus.
        debug (bool): Se True, exibe mensagens de debug e erros.

    Returns:
        tuple: Quantidade de imagens que passaram e que não passaram no teste.
    """

    # Criar as pastas de saída se não existirem
    create_output_folders(output_base_folder)

    # Inicializar o modelo FaceMesh
    face_mesh = initialize_face_mesh()

    passed = 0
    failed = 0

    # Processar imagens nas pastas de entrada
    for category, input_folder in input_folders.items():
        for img_name in os.listdir(input_folder):
            img_path = os.path.join(input_folder, img_name)

            # Processar a imagem
            angles = process_image(img_path, face_mesh)
            if angles is None:
                if debug:
                    print(f"Não foi possível processar a imagem: {img_name}")
                continue

            roll_deg, pitch_deg, yaw_deg = angles
            if debug:
                print(f"{category} - {img_name}: Roll: {roll_deg:.2f}, Pitch: {pitch_deg:.2f}, Yaw: {yaw_deg:.2f}")

            # Verificar se os ângulos estão abaixo do limite
            if abs(roll_deg) < angle_threshold and abs(pitch_deg) < angle_threshold and abs(yaw_deg) < angle_threshold:
                # Imagem passa no teste
                target_folder = os.path.join(output_base_folder, category)
                copy_image_to_folder(img_path, target_folder)
                passed += 1
            else:
                # Imagem não passa no teste
                target_folder = os.path.join(output_base_folder, f"Remove_{category}")
                copy_image_to_folder(img_path, target_folder)
                failed += 1

    return passed, failed

# Definir os diretórios de entrada e saída
input_folders = {
    "Autistic": "../data/raw_M-C-A/Autistic",
    "Non_Autistic": "../data/raw_M-C-A/Non_Autistic"
}
output_base_folder = "../data/raw_M-C-A-F"

# Chamar a função para processar as imagens
passed, failed = process_images_in_folder(input_folders, output_base_folder, debug=True)

# Exibir resultados
print(f"Imagens que passaram: {passed}")
print(f"Imagens que não passaram: {failed}")
