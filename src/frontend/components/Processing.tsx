import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';

const Processing: React.FC = () => {
  const location = useLocation();
  const { faceMeshData, croppedImage } = location.state || {}; // Recebe os dados da imagem e landmarks
  const [prediction, setPrediction] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const processImage = async () => {
      if (!croppedImage) {
        setErrorMessage('Erro: Imagem recortada não encontrada.');
        return;
      }

      try {
        const response = await fetch('https://autismdetectwithmlapi.onrender.com/predict-autism', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: croppedImage }),
        });

        if (!response.ok) {
          throw new Error(`Erro HTTP: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
          setPrediction(data.prediction); // Atualiza a predição
        } else {
          setErrorMessage('Não foi possível processar a imagem.');
        }
      } catch (error) {
        console.error('Erro ao processar a imagem:', error);
        setErrorMessage('Ocorreu um erro. Tente novamente.');
      }
    };

    processImage();
  }, [croppedImage]);

  return (
    <div className="container fade-in">
      <h1>Processando...</h1>
      {errorMessage ? (
        <p style={{ color: 'red' }}>{errorMessage}</p>
      ) : prediction ? (
        <div>
          <h2>Resultado da Análise</h2>
          <p>Predição: {prediction}</p>
        </div>
      ) : (
        <p>Processando a imagem. Por favor, aguarde...</p>
      )}
    </div>
  );
};

export default Processing;
