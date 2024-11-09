from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf  # Para carregar o modelo
from sklearn.preprocessing import StandardScaler
import cv2
import mediapipe as mp
from PIL import Image
import io

app = Flask(__name__)
# Configurar CORS para permitir requisições do frontend em http://localhost:5173
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Inicializa a solução Face Mesh do MediaPipe
mp_face_mesh = mp.solutions.face_mesh

def detect_face_mesh(image_rgb):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(image_rgb)
        landmarks_3d = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for lm in face_landmarks.landmark:
                    height, width, _ = image_rgb.shape
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    z = lm.z
                    landmarks_3d.append((x, y, z))
        return landmarks_3d

@app.route('/extract-face-mesh', methods=['POST'])
def extract_face_mesh():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "Nenhuma imagem foi enviada."}), 400

    image_file = request.files['image']

    try:
        image = Image.open(image_file.stream)
        image_rgb = np.array(image.convert('RGB'))

        landmarks_3d = detect_face_mesh(image_rgb)

        if len(landmarks_3d) == 0:
            return jsonify({"success": False, "message": "Nenhuma face foi detectada."})

        return jsonify({"success": True, "faceMesh": landmarks_3d})

    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return jsonify({"success": False, "message": "Erro ao processar a imagem."}), 500


# Função para calcular a distância euclidiana
def calculate_euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Função para calcular distâncias antropométricas baseadas nos pontos fornecidos
def calculate_anthropometric_distances(face_landmarks):
    # Vale lembrar que, na API os numeros dos pontos são -1 a menos do que no codigo .py padrão, no outro código, usa-se Xi para identificar o np.array.
    # Aqui, , aqui tem-se as posições começando em 0.
    # Tamanhos da face
    face_width_ref1 = np.array(face_landmarks[126])
    face_width_ref2 = np.array(face_landmarks[355])
    
    # Parte Superior do rosto
    trichion = np.array(face_landmarks[9])
    glabella = np.array(face_landmarks[8])
    frontozygomaticus_left = np.array(face_landmarks[299])
    frontozygomaticus_right = np.array(face_landmarks[69])
    
    # Olhos
    endo_canthus_left = np.array(face_landmarks[132])
    endo_canthus_right = np.array(face_landmarks[361])
    exo_canthus_left = np.array(face_landmarks[262])
    exo_canthus_right = np.array(face_landmarks[32])
    
    # Nariz
    upper_philtrum = np.array(face_landmarks[18])
    alare_left = np.array(face_landmarks[293])
    alare_right = np.array(face_landmarks[63])
    
    # Labios
    lower_philtrum = np.array(face_landmarks[0])
    christa_philtri_left = np.array(face_landmarks[266])
    christa_philtri_right = np.array(face_landmarks[36])
    cheilion_left = np.array(face_landmarks[60])
    cheilion_right = np.array(face_landmarks[290])
    
    # Queixo
    pogonion = np.array(face_landmarks[198])
    menton = np.array(face_landmarks[151])

    # Cálculos das distâncias antropométricas
    distances = {
        "upper_facial_height": calculate_euclidean_distance(trichion, glabella),
        "middle_facial_height": calculate_euclidean_distance(glabella, menton),
        "intercanthal_width": calculate_euclidean_distance(endo_canthus_left, endo_canthus_right),
        "biocular_width": calculate_euclidean_distance(exo_canthus_left, exo_canthus_right),
        "nasal_width": calculate_euclidean_distance(alare_left, alare_right),
        "mouth_width": calculate_euclidean_distance(cheilion_left, cheilion_right),
        "philtrum_height": calculate_euclidean_distance(upper_philtrum, lower_philtrum),
        
        # Distâncias sugeridas por artigos
        "eye_left_width": calculate_euclidean_distance(exo_canthus_left, endo_canthus_left),
        "eye_right_width": calculate_euclidean_distance(endo_canthus_right, exo_canthus_right),
        "endo_canthus_glabella_left": calculate_euclidean_distance(endo_canthus_left, glabella),
        "endo_canthus_glabella_right": calculate_euclidean_distance(glabella, endo_canthus_right),
        "exo_canthus_christa_philtri_left": calculate_euclidean_distance(christa_philtri_left, exo_canthus_left),
        "exo_canthus_christa_philtri_right": calculate_euclidean_distance(exo_canthus_right, christa_philtri_right),
        "alare_left_lower_philtrum": calculate_euclidean_distance(alare_left, lower_philtrum),
        "glabella_alare_right": calculate_euclidean_distance(glabella, alare_right),
        "glabella_christa_philtri_left": calculate_euclidean_distance(glabella, christa_philtri_left),
        "glabella_lower_philtrum": calculate_euclidean_distance(glabella, lower_philtrum),
        "glabella_christa_philtri_right": calculate_euclidean_distance(glabella, christa_philtri_right),
        "christa_philtri_right_alare_left": calculate_euclidean_distance(alare_left, christa_philtri_right),
        "christa_philtri_right_cheilion_left": calculate_euclidean_distance(cheilion_left, christa_philtri_right),
        "christa_philtri_left_cheilion_right": calculate_euclidean_distance(christa_philtri_left, cheilion_right),
        "christa_philtri_left_lower_philtrum": calculate_euclidean_distance(lower_philtrum, christa_philtri_left),
        "cheilion_left_lower_philtrum": calculate_euclidean_distance(cheilion_left, lower_philtrum),
        "cheilion_left_christa_philtri_right": calculate_euclidean_distance(cheilion_left, christa_philtri_right),
        "cheilion_left_cheilion_right": calculate_euclidean_distance(cheilion_left, cheilion_right),
        "cheilion_left_pogonion": calculate_euclidean_distance(cheilion_left, pogonion),
        "cheilion_right_lower_philtrum": calculate_euclidean_distance(cheilion_right, lower_philtrum),
        "cheilion_right_christa_philtri_right": calculate_euclidean_distance(cheilion_right, christa_philtri_right),
        "frontozygomaticus_endo_cantus_left": calculate_euclidean_distance(frontozygomaticus_left, exo_canthus_left),
        "frontozygomaticus_left_alare_right": calculate_euclidean_distance(frontozygomaticus_left, alare_right),
        "frontozygomaticus_left_cheilion_right": calculate_euclidean_distance(frontozygomaticus_left, cheilion_right),
        "frontozygomaticus_endo_cantus_right": calculate_euclidean_distance(frontozygomaticus_right, endo_canthus_right),
        "frontozygomaticus_right_cheilion_left": calculate_euclidean_distance(frontozygomaticus_right, cheilion_left),
        "face_height": calculate_euclidean_distance(trichion, menton),
        "face_width": calculate_euclidean_distance(face_width_ref2, face_width_ref1),
        # Adicionando mais quatro features para alcançar 39
        "lower_face_height": calculate_euclidean_distance(menton, pogonion),
        "eye_to_mouth_left": calculate_euclidean_distance(exo_canthus_left, cheilion_left),
        "eye_to_mouth_right": calculate_euclidean_distance(exo_canthus_right, cheilion_right),
        "nose_to_menton": calculate_euclidean_distance(upper_philtrum, menton)
    }

    return distances

# Função para preparar os dados para o modelo
def prepare_data_for_model(anthropometric_data):
    features = np.array(list(anthropometric_data.values())).reshape(1, -1)

    # Normalizar os dados
    #scaler = StandardScaler()
    #features_scaled = scaler.fit_transform(features)

    #return features_scaled
    return features
    
# Carregar o modelo salvo
model_path = '../models/best_model_3.0_layers_2_neurons_32_lr_0.001_epochs_30.h5'
model = tf.keras.models.load_model(model_path)

@app.route('/predict-autism', methods=['POST'])
def predict_autism():
    data = request.get_json()

    if 'faceMesh' not in data:
        return jsonify({"success": False, "message": "Os dados de faceMesh não foram enviados."}), 400

    face_landmarks = data['faceMesh']

    # Calcular as distâncias antropométricas
    anthropometric_distances = calculate_anthropometric_distances(face_landmarks)
    
    if len(anthropometric_distances) != 39:
        return jsonify({"success": False, "message": "Número incorreto de features calculadas."}), 400

    # Preparar os dados para o modelo
    features = prepare_data_for_model(anthropometric_distances)
    
    # Fazer a predição
    prediction = model.predict(features)
    predicted_class = int(np.round(prediction[0][0]))  # 0 ou 1

    return jsonify({
        "success": True,
        "prediction": predicted_class,
        "confidence": float(prediction[0][0])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)   
