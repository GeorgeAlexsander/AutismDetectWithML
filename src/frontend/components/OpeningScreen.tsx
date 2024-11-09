import React from 'react';
import { useNavigate } from 'react-router-dom';

const OpeningScreen: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="container fade-in opening-screen">
      <h1>Auxiliar de Diagnóstico de Autismo</h1>
      {/* Caminho correto da imagem, usando process.env.PUBLIC_URL */}
      <img 
        src={`${process.env.PUBLIC_URL}/assets/autism_heart_symbol.png`} 
        alt="Imagem ilustrativa" 
        style={{ maxWidth: '80%', margin: '20px 0' }} 
      />
      <button id="continue-btn" onClick={() => navigate('/instructions')}>
        Começar
      </button>
    </div>
  );
};

export default OpeningScreen;
