import { Fragment, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import {
  actionBadgeClass, friendlyAgent, friendlyFieldName, resultInfo,
} from '../format.js';
import { Badge, FieldLabel, Panel, PanelHeader, selectClass } from '../ui.jsx';

const EVIDENCE_BADGE = {
  CLEAN: 'border-brand-green/40 bg-brand-green/10 text-brand-green',
  SUSPICIOUS: 'border-brand-yellow/40 bg-brand-yellow/10 text-brand-yellow',
  CONFIRMED: 'border-brand-red/40 bg-brand-red/10 text-brand-red',
  VERIFY: 'border-brand-orange/40 bg-brand-orange/10 text-brand-orange',
  UNKNOWN: 'border-line-strong bg-surface-2 text-muted',
};

function EvidenceBadge({ value }) {
  const evidence = String(value || 'UNKNOWN').toUpperCase();
  return (
    <Badge className={EVIDENCE_BADGE[evidence] || EVIDENCE_BADGE.UNKNOWN}>
      {evidence.charAt(0) + evidence.slice(1).toLowerCase()}
    </Badge>
  );
}

function ActionBadge({ action }) {
  return <Badge className={actionBadgeClass(action)}>{action || '—'}</Badge>;
}

function ObservationValue({ value, depth = 0 }) {
  if (value === null || value === undefined || value === '') {
    return <span className="italic text-dim">Not observed</span>;
  }
  if (Array.isArray(value)) {
    if (!value.length) return <span className="italic text-dim">None observed</span>;
    return (
      <div className="space-y-1.5">
        {value.map((item, index) => (
          <div key={index} className="rounded-lg border border-line bg-surface px-2.5 py-1.5">
            {item && typeof item === 'object' && depth < 4 ? (
              <Fields entries={Object.entries(item)} depth={depth + 1} />
            ) : (
              <span>{item === null ? 'Not observed' : String(item)}</span>
            )}
          </div>
        ))}
      </div>
    );
  }
  if (typeof value === 'object' && depth < 4) {
    return <Fields entries={Object.entries(value)} depth={depth + 1} />;
  }
  return <span>{typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value)}</span>;
}

function Fields({ entries, depth }) {
  return (
    <dl className="grid gap-1.5 sm:grid-cols-[minmax(120px,190px)_1fr]">
      {entries.map(([key, value]) => (
        <Fragment key={key}>
          <dt className="text-[11px] font-medium uppercase tracking-wide text-dim">{friendlyFieldName(key)}</dt>
          <dd className="min-w-0 break-words text-xs text-ink">
            <ObservationValue value={value} depth={depth} />
          </dd>
        </Fragment>
      ))}
    </dl>
  );
}

function ObservationSection({ title, value, defaultOpen }) {
  const [open, setOpen] = useState(Boolean(defaultOpen));
  return (
    <div className="overflow-hidden rounded-xl border border-line bg-surface-2">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-xs font-semibold text-ink hover:bg-surface-hover"
        aria-expanded={open}
      >
        {title}
        <ChevronDown size={14} className={`shrink-0 text-dim transition ${open ? 'rotate-180' : ''}`} />
      </button>
      {open ? (
        <div className="border-t border-line px-3 py-2.5">
          <ObservationValue value={value} />
        </div>
      ) : null}
    </div>
  );
}

function Metric({ label, value, tone }) {
  return (
    <div className="rounded-xl border border-line bg-surface-2 px-3 py-2">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-dim">{label}</p>
      <p className={`mt-0.5 font-mono text-lg font-bold ${tone}`}>{value}</p>
    </div>
  );
}

export default function AnalystView({ blueView }) {
  const [agentName, setAgentName] = useState('');
  const [hostName, setHostName] = useState('');
  const [streamAgent, setStreamAgent] = useState('all');
  const [showSleep, setShowSleep] = useState(false);

  const agents = blueView?.agents || [];
  const trace = blueView?.blue_trace || [];
  const agent = agents.find((item) => item.name === agentName) || agents[0];
  const hosts = agent?.hosts || [];
  const host = hosts.find((item) => item.name === hostName) || hosts[0];
  const stream = trace
    .filter((entry) => streamAgent === 'all' || entry.agent === streamAgent)
    .filter((entry) => showSleep || entry.action !== 'Sleep')
    .slice(0, 80);
  const hostActions = trace
    .filter((entry) => entry.agent === agent?.name && entry.target === host?.name)
    .slice(0, 6);
  const pendingHosts = agents.reduce(
    (count, item) => count + item.hosts.filter((entry) => entry.pending).length, 0,
  );
  const confirmedHosts = agents.reduce(
    (count, item) => count + item.hosts.filter(
      (entry) => String(entry.evidence).toUpperCase() === 'CONFIRMED',
    ).length, 0,
  );
  const failedActions = trace.filter(
    (entry) => String(entry.result).toLowerCase() === 'false',
  ).length;
  const scope = !blueView || blueView.status === 'idle'
    ? 'No episode launched.'
    : `${blueView.blue_policy || 'No policy'} · step ${blueView.tick || 0}/${blueView.horizon || 0}`;

  const selectAgent = (name) => {
    setAgentName(name);
    setHostName('');
  };

  return (
    <Panel className="overflow-hidden">
      <PanelHeader
        title="Blue analyst view"
        subtitle={scope}
        actions={(
          <div className="w-full sm:w-52">
            <FieldLabel htmlFor="analyst-agent">Inspecting</FieldLabel>
            <select
              id="analyst-agent"
              className={selectClass}
              value={agent?.name || ''}
              onChange={(event) => selectAgent(event.target.value)}
              disabled={!agents.length}
            >
              {agents.length ? agents.map((item) => (
                <option key={item.name} value={item.name}>{friendlyAgent(item.name)}</option>
              )) : <option value="">No agents</option>}
            </select>
          </div>
        )}
      />

      <div className="space-y-4 p-4">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Metric label="Actions" value={trace.length} tone="text-brand-blue" />
          <Metric label="Pending" value={pendingHosts} tone="text-brand-yellow" />
          <Metric label="Confirmed" value={confirmedHosts} tone="text-brand-red" />
          <Metric label="Failed" value={failedActions} tone="text-brand-orange" />
        </div>

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
          <div className="overflow-hidden rounded-xl border border-line">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-surface-2/60 px-3 py-2">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-dim">Blue action stream</h4>
              <div className="flex items-center gap-2">
                <select
                  className={`${selectClass} w-auto px-2 py-1 text-xs`}
                  value={streamAgent}
                  onChange={(event) => setStreamAgent(event.target.value)}
                  aria-label="Filter trace by agent"
                >
                  <option value="all">All agents</option>
                  {agents.map((item) => (
                    <option key={item.name} value={item.name}>{friendlyAgent(item.name)}</option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => setShowSleep((current) => !current)}
                  className={`rounded-lg border px-2 py-1 text-[10px] font-semibold ${
                    showSleep
                      ? 'border-brand-indigo/40 bg-brand-indigo/15 text-ink'
                      : 'border-line bg-surface-2 text-muted'
                  }`}
                >
                  {showSleep ? 'Hide Sleep' : 'Show Sleep'}
                </button>
              </div>
            </div>
            <div className="scrollbar-thin max-h-[30rem] overflow-auto">
              {stream.length ? stream.map((entry, index) => {
                const result = resultInfo(entry.result);
                return (
                  <button
                    key={`${entry.time}-${entry.agent}-${entry.step}-${index}`}
                    type="button"
                    onClick={() => {
                      setAgentName(entry.agent);
                      setHostName(entry.target || '');
                    }}
                    className="flex w-full items-center gap-2 border-b border-line px-3 py-2 text-left text-[10px] last:border-0 hover:bg-surface-hover"
                  >
                    <span className="w-8 shrink-0 font-mono text-dim">#{entry.step}</span>
                    <span className="w-20 shrink-0 truncate text-muted">{friendlyAgent(entry.agent)}</span>
                    <ActionBadge action={entry.action} />
                    <span className="min-w-0 flex-1 truncate font-mono text-ink">{entry.target || '—'}</span>
                    <Badge className={`shrink-0 ${result.badge}`}>{result.label}</Badge>
                  </button>
                );
              }) : (
                <p className="px-4 py-10 text-center text-xs text-dim">No matching actions yet.</p>
              )}
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex flex-wrap gap-1">
              {agents.map((item) => (
                <button
                  key={item.name}
                  type="button"
                  onClick={() => selectAgent(item.name)}
                  className={`rounded-lg border px-2 py-1 text-[10px] font-semibold ${
                    item.name === agent?.name
                      ? 'border-brand-indigo/40 bg-brand-indigo/15 text-ink'
                      : 'border-line bg-surface-2 text-muted hover:bg-surface-hover'
                  }`}
                >
                  {friendlyAgent(item.name)}
                  {item.pending_action ? (
                    <span className="ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-brand-yellow" />
                  ) : null}
                </button>
              ))}
            </div>

            <div className="scrollbar-thin max-h-44 overflow-auto rounded-xl border border-line bg-surface-2 p-1.5">
              <div className="grid grid-cols-2 gap-1">
                {hosts.map((item) => (
                  <button
                    key={item.name}
                    type="button"
                    onClick={() => setHostName(item.name)}
                    className={`min-w-0 rounded-lg border px-2 py-1.5 text-left ${
                      item.name === host?.name
                        ? 'border-brand-indigo/40 bg-brand-indigo/15'
                        : 'border-transparent hover:border-line'
                    }`}
                  >
                    <span className="flex items-center gap-1.5">
                      <span className={`h-1.5 w-1.5 shrink-0 rounded-full bg-current evidence-${String(item.evidence || 'unknown').toLowerCase()}`} />
                      <span className="truncate font-mono text-[10px] text-ink">{item.name}</span>
                    </span>
                    <span className="mt-0.5 block truncate text-[9px] text-dim">
                      {item.pending_action ? `${item.pending_action} pending`
                        : item.last_action ? `${item.last_action} · ${resultInfo(item.last_result).label}`
                          : 'No action'}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {host ? (
              <div className="rounded-xl border border-line bg-surface-2 px-3 py-3">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="min-w-0 flex-1 truncate font-mono text-xs font-semibold text-ink">{host.name}</span>
                  <EvidenceBadge value={host.evidence} />
                  {host.pending_action ? (
                    <ActionBadge action={host.pending_action} />
                  ) : host.last_action ? (
                    <ActionBadge action={host.last_action} />
                  ) : null}
                </div>
                <p className="mt-1 truncate font-mono text-[9px] text-dim">
                  {(host.subnets || []).join(', ') || 'subnet unknown'}
                </p>

                <div className="mt-3 space-y-1">
                  {hostActions.length ? hostActions.map((entry, index) => {
                    const result = resultInfo(entry.result);
                    return (
                      <div key={`${entry.step}-${entry.action}-${index}`} className="flex items-center gap-2 text-[10px]">
                        <span className="w-8 font-mono text-dim">#{entry.step}</span>
                        <ActionBadge action={entry.action} />
                        <span className="min-w-0 flex-1 truncate text-muted">{friendlyAgent(entry.agent)}</span>
                        <Badge className={result.badge}>{result.label}</Badge>
                      </div>
                    );
                  }) : (
                    <p className="text-[10px] italic text-dim">No actions on this host.</p>
                  )}
                </div>

                <div className="mt-3 border-t border-line pt-2">
                  <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-dim">Retained observation</p>
                  {!Object.keys(host.observation || {}).length ? (
                    <p className="text-[10px] italic text-dim">No fields observed yet.</p>
                  ) : (
                    <div className="space-y-1.5">
                      {Object.entries(host.observation).map(([key, value], index) => (
                        <ObservationSection
                          key={key}
                          title={friendlyFieldName(key)}
                          value={value}
                          defaultOpen={index < 1}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <p className="rounded-xl border border-line bg-surface-2 px-3 py-4 text-xs italic text-dim">No owned hosts.</p>
            )}
          </div>
        </div>
      </div>
    </Panel>
  );
}
