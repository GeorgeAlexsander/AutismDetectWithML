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
    const container = document.querySelector('.container');
    container?.classList.add('fade-out'); // Adiciona a classe fade-out na transição
    setTimeout(() => navigate('/processing'), 500); // Redireciona após a transição
  };

  return (
    <div className="container fade-in">
      <h1>Envie sua Foto</h1>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      <button onClick={handleSubmit} disabled={!selectedFile}>
        Continuar
      </button>
    </div>
  );
};

export default PhotoUpload;
