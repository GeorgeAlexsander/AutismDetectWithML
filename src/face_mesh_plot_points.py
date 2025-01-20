import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import random
import pandas as pd

# Variável para definir a quantidade de distâncias a serem usadas
qtd = 42

# Caminho do arquivo CSV com as distâncias
filtered_distances_file = "../data/preprocessed_landmark/face_mesh/data_processing/data_M-C-A/ranked_correlation_top_41.csv"

# Carregar o arquivo CSV e verificar se ele existe
try:
    distances_df = pd.read_csv(filtered_distances_file)
except FileNotFoundError:
    raise FileNotFoundError(f"O arquivo {filtered_distances_file} não foi encontrado.")

# Verificar se a coluna "Distance" existe no DataFrame
if "Distance" not in distances_df.columns:
    raise KeyError("A coluna 'Distance' não foi encontrada no arquivo CSV.")

# Verificar se `qtd` é menor ou igual ao número de linhas disponíveis
if qtd > len(distances_df):
    print(f"O arquivo contém apenas {len(distances_df)} distâncias. Ajustando `qtd` para {len(distances_df)}.")
    qtd = len(distances_df)

# Selecionar as `qtd` primeiras distâncias
distances = []
for _, row in distances_df.head(qtd).iterrows():
    dist = row["Distance"].strip()  # Remover espaços em branco
    try:
        # Extrair os índices x e y do formato "dist_x_y"
        x, y = map(int, dist.split('_')[1:])
        distances.append((x, y))
    except ValueError as e:
        print(f"Erro ao processar a distância '{dist}': {e}")

# Caminho das imagens
image_path_1 = "../data/tcc_images/0001M.jpg"
image_path_2 = "../data/tcc_images/0002M.jpg"

# Inicializando o MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Dicionário para armazenar cores associadas às distâncias
color_dict = {}

# Função para obter ou gerar cor para uma distância
def get_color_for_distance(dist):
    if dist not in color_dict:
        # Gerar cor aleatória se não existir
        color_dict[dist] = [random.randint(0, 255) for _ in range(3)]
    return color_dict[dist]

# Processar as imagens
for image_path in [image_path_1, image_path_2]:
    print(f"Processando a imagem: {image_path}")

    # Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Processar a imagem com MediaPipe
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print("Nenhum rosto detectado na imagem.")
        continue

    # Obter os landmarks detectados
    landmarks = results.multi_face_landmarks[0].landmark
    height, width, _ = image.shape

    # Converter os landmarks para coordenadas em pixels
    points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks]

    # Criar uma cópia da imagem para plotar
    image_with_vectors = image.copy()

    # Desenhar os vetores conectando os pontos (com cores baseadas nas distâncias)
    for (x, y) in distances:
        dist_name = f"dist_{x}_{y}"
        pt1 = points[x]  # Coordenada do ponto x
        pt2 = points[y]  # Coordenada do ponto y
        color = get_color_for_distance(dist_name)  # Obter a cor associada à distância
        cv2.line(image_with_vectors, pt1, pt2, color, 2)

    # Converter imagem para RGB e exibir com Matplotlib
    image_with_vectors_rgb = cv2.cvtColor(image_with_vectors, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_vectors_rgb)
    plt.axis("off")
    plt.title(f"Medidas faciais plotadas - {image_path.split('/')[-1]}")
    plt.show()

# Encerrar MediaPipe
face_mesh.close()
