import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const PhotoUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const navigate = useNavigate();

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleSubmit = () => {
    if (selectedFile) {
      navigate('/processing'); // Simula o envio e vai para a tela de processamento
    }
  };

  const handleChooseFileClick = () => {
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    if (fileInput) {
      fileInput.click();
    }
  };

  return (
    <div className="container fade-in">
      <h1>Envie sua Foto</h1>
      
      {/* Área de upload com imagem padrão */}
      <div className="upload-area">
        {selectedFile ? (
          <img
            src={URL.createObjectURL(selectedFile)}
            alt="Pré-visualização"
          />
        ) : (
          <img src="/src/frontend/assets/photo_children_default.png" alt="Exemplo de foto" />
        )}
      </div>

      {/* Botões de escolher arquivo e continuar */}
      <div className="buttons">
        {/* Botão para escolher arquivo */}
        <button id="choose-file-btn" onClick={handleChooseFileClick}>
          {selectedFile ? 'Escolher Novamente' : 'Escolher Arquivo'}
        </button>
        <input
          id="file-input"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'none' }} // Esconde o input de arquivo padrão
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
    </div>
  );
};

export default PhotoUpload;
