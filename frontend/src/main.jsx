import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './index.css';
import { DocumentProvider } from './context/DocumentContext.jsx';
import { ThemeProvider } from './context/ThemeContext.jsx';
import { SettingsProvider } from './hooks/useSettings.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider>
      <SettingsProvider>
        <DocumentProvider>
          <App />
        </DocumentProvider>
      </SettingsProvider>
    </ThemeProvider>
  </StrictMode>,
);
