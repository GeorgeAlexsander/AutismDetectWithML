import React, { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Slider from 'react-slick';
import "slick-carousel/slick/slick.css"; 
import "slick-carousel/slick/slick-theme.css"; 

const distanceImage = `${process.env.PUBLIC_URL}/assets/instructions_distance.png`;
const photoExampleImage = `${process.env.PUBLIC_URL}/assets/instructions_photo_example.png`;

const InstructionsScreen: React.FC = () => {
  const navigate = useNavigate();
  const sliderRef = useRef<Slider>(null);
  const [currentSlide, setCurrentSlide] = useState(0);

  const settings = {
    dots: true,
    infinite: false,
    speed: 500,
    afterChange: (current: number) => setCurrentSlide(current),
  };

  const handleContinue = () => {
    const slides = React.Children.toArray(sliderRef.current?.props.children);
    const slideCount = slides.length;

    if (currentSlide < slideCount - 1) {
      sliderRef.current?.slickNext();
    } else {
      navigate('/upload');
    }
  };

  return (
    <div className="container fade-in instructions-screen">
      <h1>Instruções de Uso</h1>
      <Slider ref={sliderRef} {...settings}>
        <div className="carousel-slide">
          <p className="instruction-text">
            Tire a foto <b>a aproximadamente 0.5 metros</b> de distância e 
            posicione a <b>câmera na altura do rosto</b>.
          </p>
          <img src={distanceImage} alt="Exemplo de foto" className="instruction-image" />
        </div>
        <div className="carousel-slide">
          <p className="instruction-text">Exemplo de como a foto deve ficar.</p>
          <img src={photoExampleImage} alt="Exemplo de foto" className="instruction-image" />
        </div>
        <div className="carousel-slide">
          <p className="instruction-text">
           <b>Evite acessórios</b> (óculos, chapéus) que possam obstruir o rosto.
           Mantenha uma <b>expressão neutra</b> com a <b>cabeça alinhada</b>. 
          </p>
          <p className="instruction-text">
          Esta análise serve apenas como uma referência e <b>não</b> substitui a consulta com um profissional qualificado. Para informações  
          mais detalhadas, busque a orientação de um especialista.
          </p>
        </div>
      </Slider>
      <button id="prosseguir-btn" onClick={handleContinue}>Prosseguir</button>
    </div>
  );
};

export default InstructionsScreen;
