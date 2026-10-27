import { Download } from 'lucide-react';
import { EVENT_TYPE, formatTime, friendlyAgent, friendlyResult } from '../format.js';
import { Badge, Panel, PanelHeader, SecondaryButton } from '../ui.jsx';

const FILTERS = [
  { value: 'all', label: 'All' },
  { value: 'defense', label: 'Blue actions' },
  { value: 'evaluation', label: 'Evaluation labels' },
  { value: 'system', label: 'System' },
];

export default function LogsTable({ events, filter, onFilter, onExportSiem }) {
  const shown = (events || [])
    .filter((event) => filter === 'all' || event.type === filter)
    .slice(0, 200);

  return (
    <Panel className="overflow-hidden">
      <PanelHeader
        title="Episode log"
        subtitle="Newest first."
        actions={(
          <>
            <div className="flex flex-wrap items-center gap-1 rounded-lg border border-line bg-surface-2 p-1">
              {FILTERS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => onFilter(option.value)}
                  className={`rounded-md px-2.5 py-1.5 text-xs font-medium transition ${
                    filter === option.value
                      ? 'bg-brand-indigo/20 text-ink'
                      : 'text-muted hover:bg-surface-hover hover:text-ink'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <SecondaryButton onClick={onExportSiem} className="px-3 py-1.5 text-xs">
              <Download size={14} />
              Export SIEM NDJSON
            </SecondaryButton>
          </>
        )}
      />

      <div className="scrollbar-thin max-h-[36rem] overflow-auto">
        <table className="w-full min-w-[820px] border-collapse text-left text-sm">
          <thead className="sticky top-0 z-10 bg-surface-2 text-[11px] uppercase tracking-wider text-dim">
            <tr>
              <th className="px-4 py-3 font-semibold">Step</th>
              <th className="px-4 py-3 font-semibold">Time</th>
              <th className="px-4 py-3 font-semibold">Type</th>
              <th className="px-4 py-3 font-semibold">Blue agent</th>
              <th className="px-4 py-3 font-semibold">Host / target</th>
              <th className="px-4 py-3 font-semibold">What happened</th>
              <th className="px-4 py-3 font-semibold">Status</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((event, index) => {
              const type = EVENT_TYPE[event.type] || EVENT_TYPE.system;
              const result = friendlyResult(event);
              const target = event.target
                || (event.red_occupied_hosts !== undefined ? `${event.red_occupied_hosts} Red-occupied host(s)` : '—');
              return (
                <tr key={`${event.step}-${event.time}-${index}`} className="border-t border-line transition hover:bg-surface-hover">
                  <td className="whitespace-nowrap px-4 py-2.5 font-mono text-xs text-muted">
                    {event.step !== undefined ? `#${event.step}` : '—'}
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5 font-mono text-xs text-muted">{formatTime(event.time)}</td>
                  <td className="px-4 py-2.5">
                    <Badge className={type.badge}>{type.label}</Badge>
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5 text-xs text-ink">
                    {event.agent ? friendlyAgent(event.agent) : 'CAGE4'}
                  </td>
                  <td className="px-4 py-2.5 text-xs text-muted">{target}</td>
                  <td className="px-4 py-2.5 text-xs text-ink">{event.description || '—'}</td>
                  <td className="px-4 py-2.5">
                    <Badge className={result.badge}>{result.label}</Badge>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {!shown.length ? (
          <p className="px-4 py-10 text-center text-sm text-dim">
            No events yet. Launch a local CAGE4 episode.
          </p>
        ) : null}
      </div>
    </Panel>
  );
}
