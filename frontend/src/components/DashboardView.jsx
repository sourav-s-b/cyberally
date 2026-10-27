import { useEffect, useState } from 'react';

import {

  Activity,

  ArrowUpRight,

  BarChart3,

  Brain,

  CircleStop,

  Download,

  FileJson,


  MonitorPlay,

  Pause,

  Play,

  ScrollText,

  Search,

  Server,

  Users,

  X,

} from 'lucide-react';

import Brand from './Brand.jsx';

import ThemeToggle from './ThemeToggle.jsx';

import Topology from './Topology.jsx';

import RewardChart from './RewardChart.jsx';

import LogsTable from './LogsTable.jsx';

import AnalystView from './AnalystView.jsx';

import { Badge, PrimaryButton, SecondaryButton, Panel, PanelHeader, selectClass } from '../ui.jsx';

import { actionBadgeClass, classNames, friendlyAgent, resultInfo } from '../format.js';



const NAV = [


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

    return { title: 'Blue Analyst View', desc: 'Blue decisions, hosts and retained observations.' };

  }

  if (route === 'logs' || route === 'event-log') {

    return { title: 'Episode Logs', desc: 'Blue actions and evaluation labels.' };

  }

  return { title: 'Live Monitoring', desc: 'CAGE4 host graph, Blue actions and metrics.' };

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



function BlueTracePanel({ trace, onMore }) {

  return (

    <Panel className="h-full overflow-hidden">

      <PanelHeader

        title="Blue action trace"

        subtitle="Latest joint-step decisions."

        actions={(

          <SecondaryButton onClick={onMore} className="px-3 py-1.5 text-xs">

            More details

            <ArrowUpRight size={14} />

          </SecondaryButton>

        )}

      />

      <div className="scrollbar-thin max-h-64 overflow-auto">

        {trace.length ? trace.slice(0, 6).map((entry, index) => {

          const result = resultInfo(entry.result);

          return (

            <div

              key={`${entry.time}-${entry.agent}-${entry.step}-${index}`}

              className="flex items-center gap-2 border-b border-line px-3 py-2 text-[10px] last:border-0"

            >

              <span className="w-8 shrink-0 font-mono text-dim">#{entry.step}</span>

              <span className="w-20 shrink-0 truncate text-muted">{friendlyAgent(entry.agent)}</span>

              <Badge className={`shrink-0 ${actionBadgeClass(entry.action)}`}>{entry.action}</Badge>

              <span className="min-w-0 flex-1 truncate font-mono text-muted">{entry.target || '—'}</span>

              <Badge className={`shrink-0 ${result.badge}`}>{result.label}</Badge>

            </div>

          );

        }) : (

          <p className="px-4 py-8 text-center text-xs text-dim">No Blue decisions yet.</p>

        )}

      </div>

    </Panel>

  );

}



function RedTracePanel({ redAgent, redLlm, trace, onMore, redAnimate, onToggleRedAnimate }) {

  const isPlanner = Boolean(redLlm);

  return (

    <Panel className="h-full overflow-hidden">

      <PanelHeader

        title={isPlanner ? 'Red planner trace' : 'Red action trace'}

        subtitle={isPlanner ? 'Local Ollama decisions.' : 'Native Red actions and outcomes.'}

        actions={(

          <>

            {isPlanner ? (

              <>

                <Badge className="border-brand-indigo/30 bg-brand-indigo/10 text-brand-indigo">

                  <Brain size={12} />

                  {redLlm.model}

                </Badge>

                <span className="font-mono text-[10px] text-dim">

                  {redLlm.requests}/{redLlm.budget}

                </span>

              </>

            ) : (

              <Badge className="border-brand-red/30 bg-brand-red/10 text-brand-red">

                {redAgent || 'Red'}

              </Badge>

            )}

            <SecondaryButton
              onClick={onToggleRedAnimate}
              className={`px-2.5 py-1.5 text-xs ${redAnimate
                ? 'border-brand-red/50 bg-brand-red/15 text-brand-red hover:bg-brand-red/20'
                : 'text-muted'}`}
              aria-pressed={redAnimate}
              aria-label={`Red action animation ${redAnimate ? 'on' : 'off'}`}
              title="Toggle Red action animation on the topology"
            >
              <Activity size={13} />
              Red animation: {redAnimate ? 'On' : 'Off'}
            </SecondaryButton>

            <SecondaryButton onClick={onMore} disabled={!trace.length} className="px-3 py-1.5 text-xs">

              More details

            </SecondaryButton>

          </>

        )}

      />

      <div className="scrollbar-thin max-h-64 overflow-auto">

        {trace.length ? trace.slice(0, 6).map((decision, index) => {

          const result = resultInfo(decision.result);

          return (

            <div

              key={`${decision.time}-${decision.agent}-${decision.step}-${index}`}

              className="border-b border-line px-3 py-2 last:border-0"

            >

              <div className="flex min-w-0 items-center gap-1.5">

                <span className="w-8 shrink-0 font-mono text-[10px] text-dim">#{decision.step}</span>

                <span className="w-20 shrink-0 truncate text-[10px] text-muted" title={decision.agent}>

                  {friendlyAgent(decision.agent)}

                </span>

                {isPlanner && decision.goal ? (

                  <Badge className="max-w-24 shrink-0 border-brand-purple/30 bg-brand-purple/10 text-brand-purple">

                    <span className="truncate">{decision.goal}</span>

                  </Badge>

                ) : null}

                <Badge className="shrink-0 border-brand-blue/30 bg-brand-blue/10 text-brand-blue">

                  {decision.action || '—'}

                </Badge>

                {isPlanner ? (

                  <Badge className={`shrink-0 ${planTone(decision.plan_status)}`}>

                    {decision.plan_status || 'tactical'}

                  </Badge>

                ) : (

                  <Badge className={`shrink-0 ${result.badge}`}>{result.label}</Badge>

                )}

                <span className="min-w-0 flex-1 truncate text-right font-mono text-[10px] text-muted">

                  {decision.focus_ip || decision.target || parameterSummary(decision.parameters)}

                </span>

              </div>

            </div>

          );

        }) : (

          <p className="px-4 py-8 text-center text-xs text-dim">

            {isPlanner ? 'Waiting for a planner decision.' : 'Waiting for a Red action.'}

          </p>

        )}

      </div>

    </Panel>

  );

}



function RedTraceDialog({ redAgent, redLlm, trace, onClose }) {

  const isPlanner = Boolean(redLlm);

  useEffect(() => {

    const onKeyDown = (event) => {

      if (event.key === 'Escape') onClose();

    };

    window.addEventListener('keydown', onKeyDown);

    return () => window.removeEventListener('keydown', onKeyDown);

  }, [onClose]);



  return (

    <div

      className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-4 backdrop-blur-sm"

      role="presentation"

      onClick={onClose}

    >

      <div

        role="dialog"

        aria-modal="true"

        aria-label="Red trace details"

        className="flex max-h-[88vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-line-strong bg-surface shadow-2xl"

        onClick={(event) => event.stopPropagation()}

      >

        <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">

          <div>

            <h3 className="text-sm font-semibold text-ink">

              {isPlanner ? 'Red planner trace details' : 'Red action trace details'}

            </h3>

            <p className="mt-1 font-mono text-[11px] text-dim">

              {isPlanner ? (

                <>

                  {redLlm.model} · requests {redLlm.requests}/{redLlm.budget}

                  {redLlm.ollama_version ? ` · Ollama ${redLlm.ollama_version}` : ''}

                </>

              ) : (redAgent || 'Native Red')}

            </p>

          </div>

          <button

            type="button"

            onClick={onClose}

            className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-line text-muted hover:bg-surface-hover hover:text-ink"

            aria-label="Close trace details"

          >

            <X size={16} />

          </button>

        </div>

        <div className="scrollbar-thin overflow-auto">

          {trace.length ? trace.map((decision, index) => {

            const result = resultInfo(decision.result);

            return (

              <div

                key={`${decision.time}-${decision.agent}-${decision.step}-${index}`}

                className="grid gap-3 border-b border-line px-5 py-4 last:border-0 sm:grid-cols-[9rem_minmax(0,1fr)]"

              >

                <div className="font-mono text-[10px] leading-relaxed text-dim">

                  <p className="text-xs font-semibold text-ink">{friendlyAgent(decision.agent)}</p>

                  <p>step {decision.step}</p>

                  {isPlanner && decision.latency_seconds !== null ? (

                    <p>{decision.latency_seconds}s</p>

                  ) : null}

                  {!isPlanner ? <p>{result.label}</p> : null}

                </div>

                <div className="min-w-0">

                  <div className="flex flex-wrap items-center gap-1.5">

                    {isPlanner ? (

                      <>

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

                        <span className="font-mono text-[10px] text-dim">

                          plan {decision.plan_attempts ?? '—'}

                        </span>

                      </>

                    ) : (

                      <>

                        <Badge className="border-brand-blue/30 bg-brand-blue/10 text-brand-blue">

                          {decision.action || '—'}

                        </Badge>

                        <Badge className={result.badge}>{result.label}</Badge>

                        {decision.target ? (

                          <Badge className="border-line-strong bg-surface-2 text-muted">

                            target {decision.target}

                          </Badge>

                        ) : null}

                      </>

                    )}

                  </div>

                  <p className="mt-2 break-words font-mono text-[10px] leading-relaxed text-muted">

                    {parameterSummary(decision.parameters)}

                  </p>

                  {decision.prompt_sha256 ? (

                    <p className="mt-1 break-all font-mono text-[9px] leading-relaxed text-dim">

                      prompt {decision.prompt_sha256}

                    </p>

                  ) : null}

                  {Object.keys(decision.parameters || {}).length ? (

                    <details className="mt-2 text-[10px] text-dim">

                      <summary className="cursor-pointer font-semibold uppercase tracking-wider">Raw parameters</summary>

                      <pre className="scrollbar-thin mt-1 max-h-48 overflow-auto rounded-lg border border-line bg-surface-2 p-2 font-mono text-[9px] leading-relaxed text-muted">

                        {JSON.stringify(decision.parameters, null, 2)}

                      </pre>

                    </details>

                  ) : null}

                </div>

              </div>

            );

          }) : (

            <p className="px-5 py-10 text-center text-sm text-dim">No Red actions recorded yet.</p>

          )}

        </div>

      </div>

    </div>

  );

}



function LiveTab({ episode, onNavigate }) {

  const [redDetailsOpen, setRedDetailsOpen] = useState(false);
  const [redAnimate, setRedAnimate] = useState(true);

  const redOccupied = episode?.evaluation_only?.red_occupied_hosts;

  const totalReward = episode?.native_team_reward_cumulative;

  const redLlm = episode?.red_llm;

  const redAgent = episode?.red_agent;

  const redTrace = episode?.red_trace || [];

  const blueTrace = episode?.blue_trace || [];



  return (

    <div className="space-y-5">

      <Panel className="overflow-hidden">

        <PanelHeader

          title="CAGE4 live topology"

          subtitle="Blue evidence by zone. The newest Red action pulses on its target host connector; action type sets its color."

          actions={(

            <div className="flex flex-wrap items-center gap-2.5">

              {LEGEND.map(([label, dot]) => (

                <span key={label} className="inline-flex items-center gap-1.5 text-[11px] text-muted">

                  <span className={`h-2 w-2 rounded-full ${dot}`} />

                  {label}

                </span>

              ))}

            </div>

          )}

        />

        <Topology
          hosts={episode?.hosts || []}
          redAnimation={redAnimate}
          redTrace={redTrace}
        />

      </Panel>



      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">

        <StatCard icon={Users} title="Red-occupied hosts" note="(evaluation only)" value={redOccupied ?? '—'} tone="text-brand-red" border="border-l-brand-red" box="bg-brand-red/10" />

        <StatCard icon={Activity} title="Blue agent decisions" value={(episode?.tick || 0) * 5} tone="text-brand-blue" border="border-l-brand-blue" box="bg-brand-blue/10" />

        <StatCard icon={BarChart3} title="Recent events" value={episode?.events?.length ?? 0} tone="text-brand-purple" border="border-l-brand-purple" box="bg-brand-purple/10" />

        <StatCard icon={Server} title="CAGE4 hosts" value={episode?.hosts?.length ?? '—'} tone="text-brand-orange" border="border-l-brand-orange" box="bg-brand-orange/10" />

      </div>



      <Panel className="overflow-hidden">

        <PanelHeader

          title="Native team reward"

          subtitle="Per-step penalty bars overlaid with the cumulative penalty line. Both scales start at zero."

          actions={(

            <span className="rounded-lg border border-line bg-surface-2 px-3 py-1.5 font-mono text-xs text-ink">

              {totalReward !== undefined ? `Cumulative ${Number(totalReward).toFixed(2)}` : '—'}

            </span>

          )}

        />

        <RewardChart history={episode?.reward_history || []} />

      </Panel>



      <div className="grid gap-4 xl:grid-cols-2">

        <RedTracePanel

          redAgent={redAgent}

          redLlm={redLlm}

          trace={redTrace}

          onMore={() => setRedDetailsOpen(true)}
          redAnimate={redAnimate}
          onToggleRedAnimate={() => setRedAnimate((enabled) => !enabled)}

        />

        <BlueTracePanel trace={blueTrace} onMore={() => onNavigate('blue-analyst')} />

      </div>



      {redDetailsOpen && redTrace.length ? (

        <RedTraceDialog

          redAgent={redAgent}

          redLlm={redLlm}

          trace={redTrace}

          onClose={() => setRedDetailsOpen(false)}

        />

      ) : null}

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

  onStop,

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

            ? `${episode.blue_policy || 'No Blue'} vs ${episode.red_agent || 'No Red'} · seed ${episode.seed ?? '—'}`

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

            <SecondaryButton

              onClick={onStop}

              disabled={!controllable}

              className="px-3 py-1.5 text-xs hover:border-brand-red/40 hover:bg-brand-red/10 hover:text-brand-red"

            >

              <CircleStop size={14} />

              Stop

            </SecondaryButton>

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

          <Brand compact />

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
