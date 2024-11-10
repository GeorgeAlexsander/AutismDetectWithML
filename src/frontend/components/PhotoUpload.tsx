// src/frontend/components/PhotoUpload.tsx

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

// Importando a imagem diretamente
import defaultPhoto from '../assets/photo_children_default.png';

const PhotoUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const navigate = useNavigate();

  // Função que lida com a seleção do arquivo
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
    }
  };

  // Função que lida com o envio da imagem para o backend
  const handleSubmit = async () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append('image', selectedFile);

      try {
        // Faz a requisição para o backend
        const response = await fetch('http://localhost:5000/extract-face-mesh', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Erro HTTP: ${response.status}`);
        }

        // Pega os dados da resposta
        const data = await response.json();

        if (data.success) {
          console.log('Face Mesh Data:', data.faceMesh); // Mostra os dados da face mesh
          // Redireciona para a página de processamento
          navigate('/processing', { state: { faceMeshData: data.faceMesh } });
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

      {/* Área de upload da imagem */}
      <div className="upload-area">
        {selectedFile ? (
          <img
            src={URL.createObjectURL(selectedFile)}
            alt="Pré-visualização"
          />
        ) : (
          <img src={defaultPhoto} alt="Exemplo de foto" />
        )}
      </div>

      {/* Botões de ação */}
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
        
        {/* Botão de continuar desativado até o upload da foto, com tooltip */}
        <button
          id="continue-btn"
          onClick={handleSubmit}
          disabled={!selectedFile}
          title={!selectedFile ? 'Envie uma foto primeiro' : ''}
        >
          Continuar
        </button>
      </div>

      {/* Mensagem de erro */}
      {errorMessage && <p style={{ color: 'red' }}>{errorMessage}</p>}
    </div>
  );
};

export default PhotoUpload;
