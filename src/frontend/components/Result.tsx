import React from 'react';

const Result: React.FC = () => {
  return (
    <div className="container result-container">
      <h1>Resultado da Análise</h1>
      <img src="/src/frontend/assets/autism_heart_symbol.png" alt="Símbolo do Coração do Autismo" />
      <p>Nosso algoritmo detectou indícios de autismo.</p>
      <p>Para uma avaliação mais completa e confiavel, procure um profissional especializado.</p>
    </div>
  );
};

export default Result;


