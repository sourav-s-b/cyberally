import { Fragment, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { friendlyFieldName } from '../format.js';
import { FieldLabel, Panel, PanelHeader, selectClass } from '../ui.jsx';

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

export default function AnalystView({ blueView }) {
  const [agentName, setAgentName] = useState('');
  const [hostName, setHostName] = useState('');

  const agents = blueView?.agents || [];
  const agent = agents.find((item) => item.name === agentName) || agents[0];
  const hosts = agent?.hosts || [];
  const host = hosts.find((item) => item.name === hostName) || hosts[0];
  const scope = blueView
    ? `${blueView.status || 'idle'} · step ${blueView.tick || 0}/${blueView.horizon || 0} · ${blueView.view_scope || 'Blue view only'}`
    : 'Launch a CAGE4 episode to inspect Blue-visible observations.';

  return (
    <Panel className="overflow-hidden">
      <PanelHeader
        title="Blue analyst view"
        subtitle={scope}
        actions={(
          <div className="w-full sm:w-48">
            <FieldLabel htmlFor="analyst-agent">Blue agent</FieldLabel>
            <select
              id="analyst-agent"
              className={selectClass}
              value={agent?.name || ''}
              onChange={(event) => {
                setAgentName(event.target.value);
                setHostName('');
              }}
              disabled={!agents.length}
            >
              {agents.length ? agents.map((item) => (
                <option key={item.name} value={item.name}>{item.name}</option>
              )) : <option value="">No agents</option>}
            </select>
          </div>
        )}
      />

      <div className="grid gap-4 p-4 lg:grid-cols-[250px_1fr]">
        <aside className="space-y-3">
          <div>
            <FieldLabel htmlFor="analyst-host">Owned host</FieldLabel>
            <select
              id="analyst-host"
              className={selectClass}
              value={host?.name || ''}
              onChange={(event) => setHostName(event.target.value)}
              disabled={!hosts.length}
            >
              {hosts.length ? hosts.map((item) => (
                <option key={item.name} value={item.name}>{item.name}</option>
              )) : <option value="">No hosts</option>}
            </select>
          </div>
          <div className="rounded-xl border border-line bg-surface-2 px-3 py-2.5 text-xs leading-relaxed text-muted">
            {host
              ? `Evidence: ${host.evidence} · ${host.pending ? 'action pending' : 'no action pending'} · subnet: ${(host.subnets || []).join(', ') || 'unknown'}`
              : 'No Blue-owned hosts are available.'}
          </div>
          <p className="rounded-xl border border-line bg-surface-2 px-3 py-2.5 text-[11px] leading-relaxed text-dim">
            Only Blue-visible observations are shown here. Privileged simulator state stays in the
            separate evaluation labels.
          </p>
        </aside>

        <div className="min-w-0">
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-dim">Retained Blue observation</h4>
          {!host || !Object.keys(host.observation || {}).length ? (
            <p className="rounded-xl border border-line bg-surface-2 px-3 py-4 text-sm italic text-dim">
              No detailed fields are visible yet. Blue may need to observe or analyse this host.
            </p>
          ) : (
            <div className="space-y-2">
              {Object.entries(host.observation).map(([key, value], index) => (
                <ObservationSection
                  key={key}
                  title={friendlyFieldName(key)}
                  value={value}
                  defaultOpen={index < 2}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </Panel>
  );
}
