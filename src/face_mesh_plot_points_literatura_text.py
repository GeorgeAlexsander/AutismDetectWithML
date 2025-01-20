import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Inicializar o Face Mesh do MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def process_and_plot_points_with_legend(image_path):
    """
    Processa a imagem, identifica os pontos faciais e plota os pontos com siglas correspondentes.
    Adiciona uma legenda com siglas e nomes dos pontos no lado direito.
    """
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
            "ZR": ("Zigion Right", points[127]),
            "ZL": ("Zigion Left", points[356]),
            "TR": ("Trichion", points[10]),
            "GL": ("Glabella", points[9]),
            "FZL": ("Frontozygomaticus Left", points[300]),
            "FZR": ("Frontozygomaticus Right", points[70]),
            "ECL": ("Endo Canthus Left", points[133]),
            "ECR": ("Endo Canthus Right", points[362]),
            "EXL": ("Exo Canthus Left", points[263]),
            "EXR": ("Exo Canthus Right", points[33]),
            "UP": ("Upper Philtrum", points[19]),
            "AL": ("Alare Left", points[294]),
            "AR": ("Alare Right", points[64]),
            "LP": ("Lower Philtrum", points[0]),
            "CPL": ("Christa Philtri Left", points[267]),
            "CPR": ("Christa Philtri Right", points[37]),
            "CL": ("Cheilion Left", points[61]),
            "CR": ("Cheilion Right", points[291]),
            "PG": ("Pogonion", points[199]),
            "ME": ("Menton", points[152]),
        }

        # Criar uma cópia da imagem para desenhar os pontos e as siglas
        image_with_points = image.copy()

        # Desenhar cada ponto e adicionar a sigla correspondente
        for sigla, (name, coord) in points_named.items():
            # Desenhar o ponto (círculo menor)
            cv2.circle(image_with_points, coord, 2, (80, 127, 255), -1)  # Coral color

            # Ajustar a posição do texto
            font_scale = 0.35
            font_thickness = 1
            text_size = cv2.getTextSize(sigla, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = coord[0] - text_size[0] // 2

            #Posicionar o texto
            if sigla in ["CPL"]:
                text_y = coord[1] - 6  # Coloca o texto abaixo do ponto
                text_x = coord[0]  # coloca o texto para esquerda (direita da img)
            elif sigla in ["CPR"]:
                text_y = coord[1] - 6  # Coloca o texto abaixo do ponto
                text_x = coord[0] - 14  # coloca o texto para esquerda (direita da img)
            elif sigla in ["ME", "LP"]:
                text_y = coord[1] + text_size[1] + 6  # Coloca o texto abaixo do ponto
            else:
                text_y = coord[1] - 6  # Coloca o texto acima do ponto


            # Adicionar o texto na imagem
            cv2.putText(
                image_with_points,
                sigla,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 20, 255),  # Vermelho
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        # Converter a imagem para RGB
        image_with_points_rgb = cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB)

        # Criar o layout com dois subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))

        # Exibir a imagem no lado esquerdo
        axs[0].imshow(image_with_points_rgb)
        axs[0].axis("off")
        axs[0].set_title("Pontos faciais de referência", fontsize=14)

        # Adicionar a legenda no lado direito
        axs[1].axis("off")
        axs[1].set_title("Legenda", fontsize=14)
        legend_text = "\n".join([f"{sigla}: {name}" for sigla, (name, _) in points_named.items()])
        axs[1].text(0.1, 0.5, legend_text, fontsize=12, va="center", ha="left", family="monospace")

        plt.tight_layout()
        plt.show()
    else:
        print("Nenhum rosto foi detectado na imagem.")

# Caminho para a imagem
image_path = "../data/tcc_images/0002T.jpg"

# Processar e plotar os pontos na imagem
process_and_plot_points_with_legend(image_path)
