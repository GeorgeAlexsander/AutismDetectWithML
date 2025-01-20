// src/frontend/components/Result.tsx

import React from 'react';
import { useLocation } from 'react-router-dom';

// Importando a imagem diretamente
import autismHeartSymbol from '../assets/autism_heart_symbol.png';

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
          <img src={autismHeartSymbol} alt="Símbolo do Coração do Autismo" />

          {/* Verifica o valor da predição */}
          {prediction === 1 ? (
            <p style={{ textAlign: 'justify' }}>
              Foram identificados sinais de autismo na imagem. No entanto, essa é apenas uma triagem inicial.
            </p>
          ) : (
            <p style={{ textAlign: 'justify' }}>
              Não foram detectados indícios de autismo na imagem, com a ressalva de que a avaliação do aplicativo não substitui uma análise clínica profissional.
            </p>
          )}

          <p style={{ textAlign: 'justify' }}>
            Para uma avaliação mais completa e confiável, procure um profissional especializado.
          </p>
        </>
      )}
    </div>
  );
};

export default Result;
