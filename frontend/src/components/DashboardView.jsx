import { useState } from 'react';
import {
  Activity,
  BarChart3,
  Brain,
  Download,
  FileJson,
  Home,
  MonitorPlay,
  Pause,
  Play,
  ScrollText,
  Search,
  Server,
  Users,
} from 'lucide-react';
import Brand from './Brand.jsx';
import ThemeToggle from './ThemeToggle.jsx';
import Topology from './Topology.jsx';
import RewardChart from './RewardChart.jsx';
import LogsTable from './LogsTable.jsx';
import AnalystView from './AnalystView.jsx';
import { Badge, PrimaryButton, SecondaryButton, Panel, PanelHeader, PanelNote, selectClass } from '../ui.jsx';
import { classNames } from '../format.js';

const NAV = [
  { key: 'home', label: 'Home', icon: Home },
  { key: 'live-monitoring', label: 'Live Monitoring', icon: MonitorPlay },
  { key: 'logs', label: 'Logs', icon: ScrollText },
  { key: 'blue-analyst', label: 'Blue Analyst View', icon: Search },
];

const LEGEND = [
  ['Clean', 'bg-brand-green'],
  ['Suspicious', 'bg-brand-yellow'],
  ['Confirmed', 'bg-brand-red'],
  ['Verify', 'bg-brand-orange'],
  ['Unknown', 'bg-dim'],
];

function titleFor(route) {
  if (route === 'blue-analyst') {
    return { title: 'Blue Analyst View', desc: 'Per-agent Blue observations and evidence tracker state.' };
  }
  if (route === 'logs' || route === 'event-log') {
    return { title: 'Episode Logs', desc: 'Readable history of Blue actions, evaluation labels and system messages.' };
  }
  return { title: 'Live Monitoring', desc: 'Live CAGE4 host/subnet graph, Blue actions and simulator metrics.' };
}

function StatCard({ icon: Icon, title, note, value, tone, border, box }) {
  return (
    <div className={`rounded-2xl border border-l-4 border-line bg-surface p-4 shadow-xl shadow-black/10 ${border}`}>
      <div className="flex items-start gap-3">
        <div className={`grid h-11 w-11 shrink-0 place-items-center rounded-xl ${box} ${tone}`}>
          <Icon size={21} />
        </div>
        <div className="min-w-0">
          <p className="text-xs font-semibold leading-tight text-muted">
            {title}
            {note ? <span className="ml-1 text-[10px] font-normal text-dim">{note}</span> : null}
          </p>
          <p className={`mt-1 font-mono text-2xl font-bold tracking-tight ${tone}`}>{value}</p>
        </div>
      </div>
    </div>
  );
}

function planTone(status) {
  const value = String(status || '').toLowerCase();
  if (value === 'accepted') return 'border-brand-green/30 bg-brand-green/10 text-brand-green';
  if (value === 'pending_action') return 'border-brand-yellow/30 bg-brand-yellow/10 text-brand-yellow';
  if (value === 'not_due') return 'border-line bg-surface-2 text-muted';
  return 'border-brand-red/30 bg-brand-red/10 text-brand-red';
}

const NOISY_PARAMETERS = new Set([
  'name', 'priority', 'duration', 'logs', 'dropped', 'blocked',
  'bandwidth_usage', 'route', 'route_designated', 'detection_rate', 'agent',
]);

function parameterSummary(parameters) {
  const entries = Object.entries(parameters || {}).filter(([key, value]) => (
    !NOISY_PARAMETERS.has(key)
    && !(typeof value === 'string' && value.includes(' object at '))
  ));
  if (!entries.length) return 'No target parameters';
  return entries
    .map(([key, value]) => `${key}=${value && typeof value === 'object' ? JSON.stringify(value) : value}`)
    .join(' · ');
}

function LiveTab({ episode, onNavigate }) {
  const redOccupied = episode?.evaluation_only?.red_occupied_hosts;
  const totalReward = episode?.native_team_reward_cumulative;
  const redLlm = episode?.red_llm;
  const redTrace = episode?.red_trace || [];

  return (
    <div className="space-y-5">
      <Panel className="overflow-hidden">
        <PanelHeader
          title="CAGE4 live topology"
          subtitle="Blue-visible evidence across the enterprise zones. Hosts with a pending Blue action pulse until the simulator returns a result."
          actions={(
            <div className="flex flex-wrap items-center gap-2.5">
              {LEGEND.map(([label, dot]) => (
                <span key={label} className="inline-flex items-center gap-1.5 text-[11px] text-muted">
                  <span className={`h-2 w-2 rounded-full ${dot}`} />
                  {label}
                </span>
              ))}
              <span className="text-[10px] text-dim">Hover a host for its name and status.</span>
            </div>
          )}
        />
        <Topology hosts={episode?.hosts || []} />
      </Panel>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard icon={Users} title="Red-occupied hosts" note="(evaluation only)" value={redOccupied ?? '—'} tone="text-brand-red" border="border-l-brand-red" box="bg-brand-red/10" />
        <StatCard icon={Activity} title="Blue agent decisions" value={(episode?.tick || 0) * 5} tone="text-brand-blue" border="border-l-brand-blue" box="bg-brand-blue/10" />
        <StatCard icon={BarChart3} title="Recent events" note="(max 500)" value={episode?.events?.length ?? 0} tone="text-brand-purple" border="border-l-brand-purple" box="bg-brand-purple/10" />
        <StatCard icon={Server} title="CAGE4 hosts" value={episode?.hosts?.length ?? '—'} tone="text-brand-orange" border="border-l-brand-orange" box="bg-brand-orange/10" />
      </div>

      <Panel className="overflow-hidden">
        <PanelHeader
          title="Native team reward over steps"
          subtitle="Cumulative reward reported by the CAGE4 simulator. Use the Speed control to slow the episode down."
          actions={(
            <span className="rounded-lg border border-line bg-surface-2 px-3 py-1.5 font-mono text-xs text-ink">
              {totalReward !== undefined ? `Total ${Number(totalReward).toFixed(2)}` : '—'}
            </span>
          )}
        />
        <RewardChart history={episode?.reward_history || []} />
      </Panel>

      {redLlm ? (
        <Panel className="overflow-hidden">
          <PanelHeader
            title="LLM Red planner / action trace"
            subtitle="Red-only planner decisions from local Ollama. The trace shows the validated goal, optional focus and the native tactical action; it never enters Blue observations."
            actions={(
              <>
                <Badge className="border-brand-indigo/30 bg-brand-indigo/10 text-brand-indigo">
                  <Brain size={12} />
                  {redLlm.model}
                </Badge>
                <span className="rounded-lg border border-line bg-surface-2 px-2.5 py-1 font-mono text-[11px] text-muted">
                  requests {redLlm.requests}/{redLlm.budget}
                </span>
              </>
            )}
          />
          <div className="scrollbar-thin max-h-[30rem] overflow-auto">
            {redTrace.length ? redTrace.slice(0, 24).map((decision, index) => (
              <div
                key={`${decision.time}-${decision.agent}-${decision.step}-${index}`}
                className="grid gap-2 border-b border-line px-4 py-3 last:border-0 sm:grid-cols-[11rem_minmax(0,1fr)]"
              >
                <div>
                  <p className="font-mono text-xs font-semibold text-ink">{decision.agent}</p>
                  <p className="mt-0.5 font-mono text-[10px] text-dim">
                    step {decision.step} · {decision.latency_seconds ?? '—'}s
                  </p>
                </div>
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <Badge className="border-brand-purple/30 bg-brand-purple/10 text-brand-purple">
                      goal {decision.goal || '—'}
                    </Badge>
                    <Badge className="border-brand-blue/30 bg-brand-blue/10 text-brand-blue">
                      {decision.action || '—'}
                    </Badge>
                    <Badge className={planTone(decision.plan_status)}>
                      {decision.plan_status || 'tactical'}
                    </Badge>
                    {decision.focus_ip ? (
                      <Badge className="border-line-strong bg-surface-2 text-muted">
                        focus {decision.focus_ip}
                      </Badge>
                    ) : null}
                  </div>
                  <p className="mt-1.5 break-words font-mono text-[10px] leading-relaxed text-muted">
                    {parameterSummary(decision.parameters)}
                  </p>
                  {decision.prompt_sha256 ? (
                    <p className="mt-1 truncate font-mono text-[9px] text-dim">
                      prompt {decision.prompt_sha256.slice(0, 16)}… · plan {decision.plan_attempts ?? '—'}
                    </p>
                  ) : null}
                </div>
              </div>
            )) : (
              <p className="px-4 py-10 text-center text-sm text-dim">
                Waiting for the first planner decision. Start with a short horizon if the local model is slow.
              </p>
            )}
          </div>
        </Panel>
      ) : null}

      <PanelNote>
        <strong className="text-ink">CAGE4 scope:</strong> this dashboard shows CybORG simulator
        steps and Blue observations. It does not represent packets, containers, Juice Shop, DVWA, or
        real network telemetry.
      </PanelNote>

      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-line bg-surface px-4 py-3 text-xs text-muted">
        <span>Every Blue action, evaluation label and simulator message is recorded.</span>
        <SecondaryButton onClick={() => onNavigate('logs')} className="px-3 py-1.5 text-xs">
          Open Logs page
          <ScrollText size={14} />
        </SecondaryButton>
      </div>
    </div>
  );
}

export default function DashboardView({
  route,
  episode,
  blueView,
  serverError,
  onNavigate,
  onTogglePause,
  onSpeed,
  onExportJson,
  onExportSiem,
  theme,
  onToggleTheme,
}) {
  const [filter, setFilter] = useState('all');
  const isLogs = route === 'logs' || route === 'event-log';
  const isAnalyst = route === 'blue-analyst';
  const isLive = !isLogs && !isAnalyst;
  const { title, desc } = titleFor(route);

  const status = episode?.status || 'idle';
  const controllable = status === 'running' || status === 'paused';
  const stepsPerSecond = status === 'running' && episode?.tick_seconds ? 1 / episode.tick_seconds : null;
  const speedValue = String(episode?.speed ?? 1);
  const speedOptions = ['0.25', '0.5', '1', '2', '4'];
  const selectValue = speedOptions.includes(speedValue) ? speedValue : '1';

  const statusTone = serverError
    ? 'border-brand-red/40 bg-brand-red/10 text-brand-red'
    : status === 'running'
      ? 'border-brand-green/40 bg-brand-green/10 text-brand-green'
      : status === 'paused'
        ? 'border-brand-yellow/40 bg-brand-yellow/10 text-brand-yellow'
        : 'border-line bg-surface-2 text-muted';

  const statusText = serverError
    ? 'CAGE4 server unavailable'
    : episode?.error || `${status[0].toUpperCase()}${status.slice(1)} · ${episode?.simulator || 'CAGE4'}`;

  const navItem = ({ key, label, icon: Icon }) => {
    const active = key === 'live-monitoring' ? isLive
      : key === 'logs' ? isLogs
        : key === 'blue-analyst' ? isAnalyst : false;
    return (
      <button
        key={key}
        type="button"
        onClick={() => onNavigate(key)}
        className={classNames(
          'flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm font-medium transition',
          active
            ? 'border border-brand-indigo/40 bg-brand-indigo/15 text-ink'
            : 'border border-transparent text-muted hover:bg-surface-hover hover:text-ink',
        )}
      >
        <Icon size={16} className={active ? 'text-brand-blue' : 'text-dim'} />
        <span className="truncate">{label}</span>
      </button>
    );
  };

  const heading = (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div className="min-w-0">
        <h1 className="text-lg font-bold tracking-tight">{title}</h1>
        <p className="mt-1 max-w-2xl text-xs leading-relaxed text-muted">
          {serverError ? serverError : episode?.scenario
            ? `${episode.scenario} · ${episode.blue_policy ? `Blue ${episode.blue_policy}` : 'No episode launched'}${episode.red_agent ? ` · Red ${episode.red_agent}` : ''}${episode.native_team_reward_cumulative !== undefined ? ` · Native team reward ${episode.native_team_reward_cumulative}` : ''}`
            : desc}
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {!isAnalyst ? (
          <>
            <div
              className={`flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-xs ${statusTone}`}
              title={serverError || statusText}
            >
              <span className={classNames('h-2 w-2 rounded-full bg-current', status === 'running' && !serverError ? 'animate-pulse' : '')} />
              <span className="max-w-44 truncate">{statusText}</span>
              <span className="font-mono text-[11px] text-muted">
                step {episode?.tick || 0}/{episode?.horizon || 0}
                {stepsPerSecond ? ` · ${stepsPerSecond.toFixed(1)}/s` : ''}
              </span>
            </div>
            <label className="flex items-center gap-1.5 text-[11px] text-dim" htmlFor="sim-speed">
              <span className="font-semibold uppercase tracking-wider">Speed</span>
              <select
                id="sim-speed"
                className={classNames(selectClass, 'w-auto px-2 py-1.5 text-xs')}
                value={selectValue}
                onChange={(event) => onSpeed(event.target.value)}
                disabled={!controllable}
              >
                <option value="0.25">0.25×</option>
                <option value="0.5">0.5×</option>
                <option value="1">1×</option>
                <option value="2">2×</option>
                <option value="4">4×</option>
              </select>
            </label>
            <PrimaryButton onClick={onTogglePause} disabled={!controllable} className="px-3 py-1.5 text-xs">
              {status === 'running' ? <Pause size={14} /> : <Play size={14} />}
              {status === 'running' ? 'Pause' : status === 'paused' ? 'Resume' : status === 'idle' ? 'Waiting' : 'Ended'}
            </PrimaryButton>
            <SecondaryButton onClick={onExportJson} className="px-3 py-1.5 text-xs">
              <FileJson size={14} />
              <span className="hidden sm:inline">Episode JSON</span>
            </SecondaryButton>
            <SecondaryButton onClick={onExportSiem} className="px-3 py-1.5 text-xs">
              <Download size={14} />
              <span className="hidden sm:inline">SIEM NDJSON</span>
            </SecondaryButton>
          </>
        ) : null}
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </div>
    </div>
  );

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col border-r border-line bg-sidebar lg:flex">
        <div className="border-b border-line px-4 py-4">
          <Brand compact onClick={() => onNavigate('home')} />
        </div>
        <nav className="flex-1 space-y-1 overflow-y-auto p-3" aria-label="Dashboard navigation">
          {NAV.map(navItem)}
        </nav>
        <div className="border-t border-line px-4 py-3 text-[10px] leading-relaxed text-dim">
          Local CybORG CAGE4 simulator.<br />
          Not a production range or SIEM.
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 border-b border-line bg-header px-4 py-3 backdrop-blur lg:px-6">
          {heading}
          <nav className="mt-3 flex gap-1 overflow-x-auto pb-0.5 lg:hidden" aria-label="Dashboard navigation">
            {NAV.map(({ key, label, icon: Icon }) => {
              const active = key === 'live-monitoring' ? isLive
                : key === 'logs' ? isLogs
                  : key === 'blue-analyst' ? isAnalyst : false;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => onNavigate(key)}
                  className={classNames(
                    'inline-flex shrink-0 items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium',
                    active
                      ? 'border-brand-indigo/40 bg-brand-indigo/15 text-ink'
                      : 'border-transparent text-muted hover:bg-surface-hover',
                  )}
                >
                  <Icon size={14} />
                  {label}
                </button>
              );
            })}
          </nav>
        </header>

        <main className="flex-1 p-4 lg:p-6">
          {isLive ? <LiveTab episode={episode} onNavigate={onNavigate} /> : null}
          {isLogs ? (
            <LogsTable
              events={episode?.events || []}
              filter={filter}
              onFilter={setFilter}
              onExportSiem={onExportSiem}
            />
          ) : null}
          {isAnalyst ? <AnalystView blueView={blueView} /> : null}
        </main>
      </div>
    </div>
  );
}
