// src/frontend/components/OpeningScreen.tsx

import React from 'react';
import { useNavigate } from 'react-router-dom';

import autismHeartSymbol from '../assets/autism_heart_symbol.png';

const OpeningScreen: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="container fade-in opening-screen">
      <h1>Auxiliar de Diagnóstico de Autismo</h1>
      <img src={autismHeartSymbol} alt="Imagem ilustrativa" style={{ maxWidth: '80%', margin: '20px 0' }} />
      <button id="continue-btn" onClick={() => navigate('/instructions')}>
        Começar
      </button>
    </div>
  );
};

export default OpeningScreen;
