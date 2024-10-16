from flask import Flask, request, jsonify
from flask_cors import CORS  # Importar CORS
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir requisições do frontend

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
