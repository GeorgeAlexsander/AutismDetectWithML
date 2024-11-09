from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf  # Para carregar o modelo
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
# Configurar CORS para permitir requisições do frontend em http://localhost:5173
CORS(app)  # Habilitar CORS para permitir requisições do frontend

# Função para calcular a distância euclidiana
def calculate_euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Função para calcular distâncias antropométricas baseadas nos pontos fornecidos
def calculate_anthropometric_distances(face_landmarks):
    # Tamanhos da face
    face_width_ref1 = np.array(face_landmarks[127])  # Acessando diretamente o ponto 127
    face_width_ref2 = np.array(face_landmarks[356])  # Acessando diretamente o ponto 356
    
    # Parte Superior do rosto
    trichion = np.array(face_landmarks[10])  # Ponto 10    
    glabella = np.array(face_landmarks[9])   # Ponto 9
    frontozygomaticus_left = np.array(face_landmarks[300])  # Ponto 300
    frontozygomaticus_right = np.array(face_landmarks[70])  # Ponto 70
    
    # Olhos
    endo_canthus_left = np.array(face_landmarks[133])  # Ponto 133
    endo_canthus_right = np.array(face_landmarks[362])  # Ponto 362
    exo_canthus_left = np.array(face_landmarks[263])  # Ponto 263
    exo_canthus_right = np.array(face_landmarks[33])  # Ponto 33
    
    # Nariz
    upper_philtrum = np.array(face_landmarks[19])  # Ponto 19
    alare_left = np.array(face_landmarks[294])  # Ponto 294
    alare_right = np.array(face_landmarks[64])  # Ponto 64
    
    # Labios
    lower_philtrum = np.array(face_landmarks[0])  # Ponto 0
    christa_philtri_left = np.array(face_landmarks[267])  # Ponto 267
    christa_philtri_right = np.array(face_landmarks[37])  # Ponto 37
    cheilion_left = np.array(face_landmarks[61])  # Ponto 61
    cheilion_right = np.array(face_landmarks[291])  # Ponto 291
    
    # Queixo
    pogonion = np.array(face_landmarks[199])  # Ponto 199
    menton = np.array(face_landmarks[152])  # Ponto 152

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
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled

# Carregar o modelo salvo
model_path = '../models/best_model_3.0_layers_2_neurons_32_lr_0.001_epochs_30.h5'
model = tf.keras.models.load_model(model_path)

# Rota para receber os dados, calcular as distâncias e fazer a predição
@app.route('/predict-autism-not-used', methods=['POST'])
def predict_autism():
    data = request.get_json()

    if 'faceMesh' not in data:
        return jsonify({"success": False, "message": "Os dados de faceMesh não foram enviados."}), 400

    face_landmarks = data['faceMesh']

    # Calcular as distâncias antropométricas
    anthropometric_distances = calculate_anthropometric_distances(face_landmarks)
    
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
