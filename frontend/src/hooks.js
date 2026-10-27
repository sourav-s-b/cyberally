import { useCallback, useEffect, useState } from 'react';

function readRoute() {
  return window.location.hash.replace(/^#\/?/, '').toLowerCase();
}

export function useHashRoute() {
  const [route, setRoute] = useState(readRoute);

  useEffect(() => {
    const onChange = () => setRoute(readRoute());
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);

  const navigate = useCallback((next) => {
    const target = `#${next}`;
    if (window.location.hash === target) setRoute(readRoute());
    else window.location.hash = target;
  }, []);

  return [route, navigate];
}

export function useTheme() {
  const [theme, setTheme] = useState(() => (
    document.documentElement.classList.contains('light-theme') ? 'light' : 'dark'
  ));

  useEffect(() => {
    document.documentElement.classList.toggle('light-theme', theme === 'light');
    try {
      localStorage.setItem('cyberally-theme', theme);
    } catch (error) {
      /* localStorage can be unavailable in strict privacy modes */
    }
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((current) => (current === 'dark' ? 'light' : 'dark'));
  }, []);

  return [theme, toggleTheme];
}

export function useToasts() {
  const [toasts, setToasts] = useState([]);

  const push = useCallback((message, kind = 'info') => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setToasts((current) => [...current.slice(-3), { id, message, kind }]);
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, 4200);
  }, []);

  return [toasts, push];
}
