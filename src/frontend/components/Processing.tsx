import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import PacmanLoader from 'react-spinners/PacmanLoader'; // Importa o PacmanLoader

const Processing: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
      navigate('/result');
    }, 3000); // Simula 3 segundos de processamento
    return () => clearTimeout(timer);
  }, [navigate]);

  return (
    <div className="container fade-in processing-container">
      <h1>Processando...</h1>
      <div className="spinner">
        <div className="pacman-wrapper">
          <PacmanLoader
            color="#1a2a6c"        // Cor do PacmanLoader
            loading={loading}       // Controla a exibição
            size={30}               // Tamanho do loader
            speedMultiplier={0.9}   // Aumenta a velocidade da animação
          />
        </div>
      </div>
      <p className="processing-text">Estamos analisando a sua foto.</p>
    </div>
  );
};

export default Processing;
