import { useCallback, useEffect, useRef, useState } from 'react';
import HomeView from './components/HomeView.jsx';
import ConfigView from './components/ConfigView.jsx';
import DashboardView from './components/DashboardView.jsx';
import Toasts from './components/Toasts.jsx';
import { downloadFile, request } from './api.js';
import { buildSiemNdjson } from './format.js';
import { useHashRoute, useTheme, useToasts } from './hooks.js';

const CONFIG_ROUTES = new Set(['config', 'configuration']);
const DASHBOARD_ROUTES = new Set(['live-monitoring', 'live', 'logs', 'event-log', 'blue-analyst']);

export default function App() {
  const [route, navigate] = useHashRoute();
  const [theme, toggleTheme] = useTheme();
  const [toasts, toast] = useToasts();
  const [models, setModels] = useState([]);
  const [modelsError, setModelsError] = useState('');
  const [modelsNotice, setModelsNotice] = useState('');
  const [episode, setEpisode] = useState(null);
  const [blueView, setBlueView] = useState(null);
  const [serverError, setServerError] = useState('');
  const [config, setConfig] = useState({
    policyKind: 'baseline',
    baseline: 'round_robin',
    modelId: '',
    redAgent: 'discovery',
    redModel: 'qwen2.5:3b',
    seed: 7629,
    horizon: 400,
  });

  const inFlight = useRef(false);
  const routeRef = useRef(route);
  routeRef.current = route;

  const view = DASHBOARD_ROUTES.has(route)
    ? 'dashboard'
    : CONFIG_ROUTES.has(route) ? 'config' : 'home';

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [view]);

  useEffect(() => {
    let cancelled = false;
    request('/api/models')
      .then(({ models: next, catalog_error: catalogError }) => {
        if (cancelled) return;
        setModels(next || []);
        setModelsNotice(catalogError || '');
        setModelsError('');
      })
      .catch((error) => {
        if (!cancelled) {
          setModelsError(error.message);
          setModelsNotice('');
        }
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (view !== 'dashboard') return undefined;
    let cancelled = false;
    const poll = async () => {
      if (inFlight.current) return;
      inFlight.current = true;
      try {
        if (routeRef.current === 'blue-analyst') {
          const data = await request('/api/blue-view');
          if (!cancelled) setBlueView(data);
        } else {
          const data = await request('/api/state');
          if (!cancelled) setEpisode(data);
        }
        if (!cancelled) setServerError('');
      } catch (error) {
        if (!cancelled) setServerError(error.message);
      } finally {
        inFlight.current = false;
      }
    };
    poll();
    const timer = window.setInterval(poll, 900);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [view]);

  const updateConfig = useCallback((patch) => {
    setConfig((current) => ({ ...current, ...patch }));
  }, []);

  const startEpisode = useCallback(async (payload) => {
    try {
      const data = await request('/api/start', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      setEpisode(data);
      setServerError('');
      navigate('live-monitoring');
      toast('Local CAGE4 episode launched', 'success');
      return true;
    } catch (error) {
      toast(`Could not start CAGE4: ${error.message}`, 'error');
      return false;
    }
  }, [navigate, toast]);

  const control = useCallback(async (body, message) => {
    try {
      const data = await request('/api/control', {
        method: 'POST',
        body: JSON.stringify(body),
      });
      setEpisode(data);
      setServerError('');
      if (message) toast(message, 'success');
    } catch (error) {
      toast(error.message, 'error');
    }
  }, [toast]);

  const togglePause = useCallback(() => {
    if (!episode || !['running', 'paused'].includes(episode.status)) return;
    control({ running: episode.status !== 'running' });
  }, [episode, control]);

  const changeSpeed = useCallback((value) => {
    if (!episode || !['running', 'paused'].includes(episode.status)) {
      toast('Launch a CAGE4 episode before changing speed.', 'info');
      return;
    }
    control(
      { running: episode.status === 'running', speed: Number(value) },
      `Simulation speed set to ${value}×`,
    );
  }, [episode, control, toast]);

  const exportJson = useCallback(async () => {
    try {
      const state = await request('/api/export');
      downloadFile(
        `cage4-episode-${state.seed ?? 'not-started'}-${Date.now()}.json`,
        JSON.stringify(state, null, 2),
        'application/json',
      );
      toast('Episode JSON export started', 'success');
    } catch (error) {
      toast(`Export failed: ${error.message}`, 'error');
    }
  }, [toast]);

  const exportSiem = useCallback(async () => {
    try {
      const state = await request('/api/export');
      if (!state.events?.length) throw new Error('No events to export yet. Launch or run an episode first.');
      downloadFile(
        `cage4-siem-${state.seed ?? 'unseeded'}-${Date.now()}.ndjson`,
        buildSiemNdjson(state),
        'application/x-ndjson',
      );
      toast('SIEM NDJSON export started (Elasticsearch _bulk format)', 'success');
    } catch (error) {
      toast(`SIEM export failed: ${error.message}`, 'error');
    }
  }, [toast]);

  return (
    <>
      {view === 'home' ? (
        <HomeView
          onStart={() => navigate('config')}
          onToast={toast}
          theme={theme}
          onToggleTheme={toggleTheme}
        />
      ) : null}

      {view === 'config' ? (
        <ConfigView
          models={models}
          modelsError={modelsError}
          modelsNotice={modelsNotice}
          config={config}
          updateConfig={updateConfig}
          onLaunch={startEpisode}
          onBack={() => navigate('home')}
          onHome={() => navigate('home')}
          onToast={toast}
          theme={theme}
          onToggleTheme={toggleTheme}
        />
      ) : null}

      {view === 'dashboard' ? (
        <DashboardView
          route={route}
          episode={episode}
          blueView={blueView}
          serverError={serverError}
          onNavigate={navigate}
          onTogglePause={togglePause}
          onSpeed={changeSpeed}
          onExportJson={exportJson}
          onExportSiem={exportSiem}
          theme={theme}
          onToggleTheme={toggleTheme}
        />
      ) : null}

      <Toasts toasts={toasts} />
    </>
  );
}
