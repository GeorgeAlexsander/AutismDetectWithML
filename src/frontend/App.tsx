// src/frontend/App.tsx
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import PhotoUpload from './components/PhotoUpload';
import Processing from './components/Processing.tsx';
import Result from './components/Result';
import './styles/styles.css';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<PhotoUpload />} />
        <Route path="/processing" element={<Processing />} />
        <Route path="/result" element={<Result />} />
      </Routes>
    </Router>
  );
};

export default App;
