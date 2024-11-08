// src/frontend/App.tsx

import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import OpeningScreen from './components/OpeningScreen';
import InstructionsScreen from './components/InstructionsScreen';
import PhotoUpload from './components/PhotoUpload';
import Processing from './components/Processing';
import Result from './components/Result';
import './styles/styles.css';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<OpeningScreen />} />
        <Route path="/instructions" element={<InstructionsScreen />} />
        <Route path="/upload" element={<PhotoUpload />} />
        <Route path="/processing" element={<Processing />} />
        <Route path="/result" element={<Result />} />
      </Routes>
    </Router>
  );
};

export default App;
