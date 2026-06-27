import { createContext, useContext, useEffect, useState } from 'react';

export const THEMES = [
  { id: 'light', label: 'Light', swatch: '#f8fafc', ring: '#2563eb' },
  { id: 'dark', label: 'Obsidian', swatch: '#121827', ring: '#10b981' },
  { id: 'midnight', label: 'Midnight', swatch: '#0f172a', ring: '#3b82f6' },
  { id: 'violet', label: 'Violet', swatch: '#1a1228', ring: '#8b5cf6' },
];

const ThemeContext = createContext(null);

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(
    () => localStorage.getItem('app-theme') || 'dark',
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('app-theme', theme);
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
  return ctx;
}
