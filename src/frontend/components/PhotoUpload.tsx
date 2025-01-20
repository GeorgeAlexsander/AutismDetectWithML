import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import defaultPhoto from '../assets/photo_children_default.png';

const PhotoUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null); // Nova variável para a imagem recortada
  const navigate = useNavigate();

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleSubmit = async () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append('image', selectedFile);

      try {
        const response = await fetch('https://autismdetectwithmlapi.onrender.com/extract-face-mesh', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Erro HTTP: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
          setProcessedImage(data.croppedImage); // Salva a imagem processada
          navigate('/processing', { 
            state: { 
              faceMeshData: data.faceMesh, 
              croppedImage: data.croppedImage // Passa os dados processados
            } 
          });
        } else {
          setErrorMessage('Não foi possível detectar um rosto na imagem. Tente novamente.');
        }
      } catch (error) {
        console.error('Erro ao enviar a imagem:', error);
        setErrorMessage('Ocorreu um erro. Tente novamente.');
      }
    }
  };

  return (
    <div className="container fade-in">
      <h1>Envie sua Foto</h1>
      <div className="upload-area">
        {selectedFile ? (
          <img src={URL.createObjectURL(selectedFile)} alt="Pré-visualização" />
        ) : (
          <img src={defaultPhoto} alt="Exemplo de foto" />
        )}
      </div>
      <div className="buttons">
        <button id="choose-file-btn" onClick={() => document.getElementById('file-input')?.click()}>
          {selectedFile ? 'Escolher Novamente' : 'Escolher Arquivo'}
        </button>
        <input
          id="file-input"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <button
          id="continue-btn"
          onClick={handleSubmit}
          disabled={!selectedFile}
          title={!selectedFile ? 'Envie uma foto primeiro' : ''}
        >
          Continuar
        </button>
      </div>
      {errorMessage && <p style={{ color: 'red' }}>{errorMessage}</p>}
    </div>
  );
};

export default PhotoUpload;
