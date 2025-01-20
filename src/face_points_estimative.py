import cv2
import mediapipe as mp
import numpy as np
import math

# Inicializar o MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def calculate_head_pose(landmarks, image):
    # Obter os pontos dos olhos
    left_eye = [landmarks[33], landmarks[133], landmarks[160], landmarks[158]]
    right_eye = [landmarks[362], landmarks[263], landmarks[249], landmarks[453]]
    
    # Calcular as distâncias horizontais e verticais
    left_eye_width = np.linalg.norm(np.array(left_eye[0]) - np.array(left_eye[2]))
    left_eye_height = np.linalg.norm(np.array(left_eye[1]) - np.array(left_eye[3]))
    
    right_eye_width = np.linalg.norm(np.array(right_eye[0]) - np.array(right_eye[2]))
    right_eye_height = np.linalg.norm(np.array(right_eye[1]) - np.array(right_eye[3]))
    
    # Calcular a diferença entre as proporções dos olhos (aspect ratio)
    left_eye_aspect_ratio = left_eye_width / left_eye_height
    right_eye_aspect_ratio = right_eye_width / right_eye_height
    
    # Calcular a diferença entre os olhos para estimar o Yaw (giro horizontal)
    eye_diff = left_eye_aspect_ratio - right_eye_aspect_ratio

    # Para Pitch, podemos verificar a distância vertical entre os olhos
    eye_vertical_distance = np.linalg.norm(np.array(left_eye[1]) - np.array(right_eye[1]))
    
    # Estimando a rotação da cabeça (Yaw, Pitch)
    yaw = eye_diff * 20  # Ajuste de fator para o yaw
    pitch = eye_vertical_distance * 5  # Ajuste de fator para o pitch
    
    # Roll estimado com base na simetria horizontal entre os olhos
    roll = (left_eye_width - right_eye_width) * 10  # Ajuste de fator para o roll
    
    return yaw, pitch, roll

# Carregar a imagem
image_path = "../data/tcc_images/0002M.jpg"
image = cv2.imread(image_path)
height, width, _ = image.shape

# Inicializar o detector do Face Mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    # Converter a imagem para RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Desenhar os marcos faciais
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            
            # Converter os marcos para um formato adequado
            landmarks = [(landmark.x * width, landmark.y * height) for landmark in face_landmarks.landmark]
            
            # Estimar a rotação da cabeça
            yaw, pitch, roll = calculate_head_pose(landmarks, image)
            
            # Imprimir os resultados no terminal
            print(f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}")
        
        # Mostrar a imagem com os marcos desenhados
        cv2.imshow("Head Pose Estimation", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
