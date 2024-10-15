import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const Processing: React.FC = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => {
      const container = document.querySelector('.container');
      container?.classList.add('fade-out');
      setTimeout(() => navigate('/result'), 500); // Redireciona após fade-out
    }, 3000); // Simula 3 segundos de processamento
    return () => clearTimeout(timer);
  }, [navigate]);

  return (
    <div className="container fade-in processing-container">
      <h1>Processando...</h1>
      <div className="animation">🔄</div>
      <p>Estamos analisando a sua foto.</p>
    </div>
  );
};

export default Processing;
