import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import PacmanLoader from 'react-spinners/PacmanLoader'; // Importa o PacmanLoader

const Processing: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [loading, setLoading] = useState(true); // Estado de carregamento

  useEffect(() => {
    // Obtém os dados de faceMesh do estado da navegação
    const faceMeshData = location.state?.faceMeshData;

    // Verificação se faceMeshData existe
    if (!faceMeshData) {
      console.error('Dados de faceMesh ausentes');
      navigate('/result', { state: { prediction: null, error: 'Dados de faceMesh ausentes.' } });
      return; // Cancela a execução se faceMeshData estiver ausente
    }

    // Função para fazer a requisição à API de predição
    const fetchPrediction = async () => {
      try {
        setLoading(true); // Ativa o estado de carregamento

        // Faz a requisição para a API no Render
        const response = await fetch('https://autismdetectwithmlapi.onrender.com/predict-autism', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ faceMesh: faceMeshData }), // Envia os dados de faceMesh para a API
        });

        // Verifica se a resposta foi bem-sucedida
        if (!response.ok) {
          throw new Error(`Erro HTTP: ${response.status}`); // Lança erro em caso de status diferente de 200
        }

        const result = await response.json();
        console.log("Resposta da API:", result);  // Adiciona log para depuração

        if (result.success) {
          // Navegar para a tela de resultado com o resultado da predição
          navigate('/result', { state: { prediction: result.prediction, confidence: result.confidence } });
        } else {
          console.error('Erro na predição:', result.message);
          navigate('/result', { state: { prediction: null, error: result.message } });
        }
      } catch (error) {
        console.error('Erro ao se comunicar com a API:', error);
        navigate('/result', { state: { prediction: null, error: 'Erro de comunicação com o servidor.' } });
      } finally {
        setLoading(false); // Desativa o estado de carregamento após o processamento
      }
    };

    // Chama a função de predição assim que a tela carrega
    fetchPrediction();

  }, [location.state, navigate]);

  return (
    <div className="container fade-in processing-container">
      <h1>Processando...</h1>
      <div className="spinner">
        <div className="pacman-wrapper">
          <PacmanLoader
            color="#1a2a6c"
            loading={loading}  // Define o loading como dependente do estado
            size={30}
            speedMultiplier={0.9}
          />
        </div>
      </div>
      <p className="processing-text">Estamos analisando a sua foto.</p>
    </div>
  );
};

export default Processing;
