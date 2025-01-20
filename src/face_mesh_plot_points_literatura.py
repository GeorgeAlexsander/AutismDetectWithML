import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import random
import numpy as np

# Caminhos das imagens
image_path_1 = "../data/tcc_images/0001M.jpg"
image_path_2 = "../data/tcc_images/0002M.jpg"

# Inicializando o MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Lista fixa de conexões entre os pontos
CONNECTIONS = [
    ("trichion", "glabella"),
    ("glabella", "menton"),
    ("endo_canthus_left", "endo_canthus_right"),
    ("exo_canthus_left", "exo_canthus_right"),
    ("alare_left", "alare_right"),
    ("cheilion_left", "cheilion_right"),
    ("upper_philtrum", "lower_philtrum"),
    ("exo_canthus_left", "endo_canthus_left"),
    ("endo_canthus_right", "exo_canthus_right"),
    ("endo_canthus_left", "glabella"),
    ("glabella", "endo_canthus_right"),
    ("christa_philtri_left", "exo_canthus_left"),
    ("exo_canthus_right", "christa_philtri_right"),
    ("alare_left", "lower_philtrum"),
    ("glabella", "alare_right"),
    ("glabella", "christa_philtri_left"),
    ("glabella", "lower_philtrum"),
    ("glabella", "christa_philtri_right"),
    ("alare_left", "christa_philtri_right"),
    ("cheilion_left", "christa_philtri_right"),
    ("lower_philtrum", "christa_philtri_right"),
    ("christa_philtri_left", "cheilion_right"),
    ("lower_philtrum", "christa_philtri_left"),
    ("cheilion_left", "lower_philtrum"),
    ("cheilion_left", "christa_philtri_right"),
    ("cheilion_left", "christa_philtri_left"),
    ("cheilion_left", "cheilion_right"),
    ("cheilion_left", "pogonion"),
    ("cheilion_right", "lower_philtrum"),
    ("cheilion_right", "christa_philtri_right"),
    ("cheilion_right", "christa_philtri_left"),
    ("frontozygomaticus_left", "exo_canthus_left"),
    ("frontozygomaticus_left", "alare_right"),
    ("frontozygomaticus_left", "cheilion_right"),
    ("frontozygomaticus_right", "endo_canthus_right"),
    ("frontozygomaticus_right", "exo_canthus_right"),
    ("frontozygomaticus_right", "cheilion_left"),
    ("trichion", "menton"),
    ("zigion_right", "zigion_left"),
    ("zigion_right", "exo_canthus_right"),
    ("zigion_left", "exo_canthus_left"),
    ("pogonion","menton")
]

# Gerar cores fixas para cada conexão
random.seed(42)  # Fixar a semente para cores consistentes
connection_colors = {connection: [random.randint(0, 255) for _ in range(3)] for connection in CONNECTIONS}

# Função para processar e plotar a imagem
def process_and_plot_image(image_path):
    # Carregar a imagem
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Processar a imagem com MediaPipe
    results = face_mesh.process(image_rgb)

    # Verificar se há landmarks detectados
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        height, width, _ = image.shape

        # Converter os landmarks para coordenadas em pixels
        points = {idx: (int(landmark.x * width), int(landmark.y * height)) for idx, landmark in enumerate(landmarks)}

        # Mapear os nomes dos pontos às coordenadas
        points_named = {
            "zigion_right": points[127],
            "zigion_left": points[356],
            "trichion": points[10],
            "glabella": points[9],
            "frontozygomaticus_left": points[300],
            "frontozygomaticus_right": points[70],
            "endo_canthus_left": points[133],
            "endo_canthus_right": points[362],
            "exo_canthus_left": points[263],
            "exo_canthus_right": points[33],
            "upper_philtrum": points[19],
            "alare_left": points[294],
            "alare_right": points[64],
            "lower_philtrum": points[0],
            "christa_philtri_left": points[267],
            "christa_philtri_right": points[37],
            "cheilion_left": points[61],
            "cheilion_right": points[291],
            "pogonion": points[199],
            "menton": points[152],
        }

        # Criar uma cópia da imagem para desenhar os vetores
        image_with_vectors = image.copy()

        # Desenhar as linhas entre os pontos selecionados
        for pt1_name, pt2_name in CONNECTIONS:
            pt1 = points_named[pt1_name]
            pt2 = points_named[pt2_name]
            color = connection_colors[(pt1_name, pt2_name)]
            cv2.line(image_with_vectors, pt1, pt2, color, 2)

        # Converter a imagem para RGB e exibir com Matplotlib
        image_with_vectors_rgb = cv2.cvtColor(image_with_vectors, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_vectors_rgb)
        plt.axis("off")
        plt.title(f"Medidas faciais plotadas: {image_path}")
        plt.show()

# Processar e plotar as duas imagens
process_and_plot_image(image_path_1)
process_and_plot_image(image_path_2)

# Encerrar MediaPipe
face_mesh.close()
