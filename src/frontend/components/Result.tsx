import React from 'react';
import { useLocation } from 'react-router-dom';

const Result: React.FC = () => {
  const location = useLocation();
  const prediction = location.state?.prediction;
  const error = location.state?.error;

  return (
    <div className="container result-container">
      <h1>Resultado da Análise</h1>

      {/* Verifica se houve erro na predição */}
      {error ? (
        <p style={{ color: 'red' }}>Ocorreu um erro: {error}</p>
      ) : (
        <>
          <img src="/src/frontend/assets/autism_heart_symbol.png" alt="Símbolo do Coração do Autismo" />

          {/* Verifica o valor da predição */}
          {prediction === 1 ? (
            <p>Nosso algoritmo detectou indícios de autismo.</p>
          ) : (
            <p>Nosso algoritmo não detectou indícios de autismo.</p>
          )}

          <p>Para uma avaliação mais completa e confiável, procure um profissional especializado.</p>
        </>
      )}
    </div>
  );
};

export default Result;
