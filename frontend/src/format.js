export const EVENT_TYPE = {
  defense: { label: 'Blue action', badge: 'border-brand-blue/30 bg-brand-blue/10 text-brand-blue' },
  evaluation: { label: 'Evaluation', badge: 'border-brand-purple/30 bg-brand-purple/10 text-brand-purple' },
  system: { label: 'System', badge: 'border-line bg-surface-2 text-muted' },
};

export function humanizeCheckpoint(model) {
  const [fallbackRun, fallbackStep = ''] = String(model.label || '').split(' · checkpoint ');
  const run = model.run_label || fallbackRun.replaceAll('_', ' ');
  const step = model.checkpoint_step || fallbackStep;
  return step ? `${run} · checkpoint ${step}` : run;
}

export function friendlyFieldName(value) {
  return String(value)
    .replaceAll('_', ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function friendlyAgent(value) {
  const blue = String(value || '').match(/^blue_agent_(\d+)$/);
  if (blue) return `Blue agent ${Number(blue[1]) + 1}`;
  const red = String(value || '').match(/^red_agent_(\d+)$/);
  if (red) return `Red agent ${Number(red[1]) + 1}`;
  return value || 'CAGE4';
}

export function resultInfo(value) {
  const result = String(value ?? '').toLowerCase();
  if (result === 'true' || result === 'success') {
    return { label: 'Success', badge: 'border-brand-green/30 bg-brand-green/10 text-brand-green' };
  }
  if (result === 'false' || result === 'failed') {
    return { label: 'Failed', badge: 'border-brand-red/30 bg-brand-red/10 text-brand-red' };
  }
  if (result === 'pending' || result === 'in_progress') {
    return { label: result === 'in_progress' ? 'In progress' : 'Pending', badge: 'border-brand-yellow/30 bg-brand-yellow/10 text-brand-yellow' };
  }
  if (result === 'issued') {
    return { label: 'Issued', badge: 'border-brand-blue/30 bg-brand-blue/10 text-brand-blue' };
  }
  return { label: value ? String(value) : 'Recorded', badge: 'border-line bg-surface-2 text-muted' };
}

export function friendlyResult(event) {
  if (event.type === 'evaluation') {
    return { label: 'Evaluation only', badge: 'border-brand-purple/30 bg-brand-purple/10 text-brand-purple' };
  }
  return resultInfo(event.result);
}

export function actionBadgeClass(action) {
  if (action === 'Remove') return 'border-brand-orange/30 bg-brand-orange/10 text-brand-orange';
  if (action === 'Restore') return 'border-brand-green/30 bg-brand-green/10 text-brand-green';
  if (action === 'Analyse') return 'border-brand-purple/30 bg-brand-purple/10 text-brand-purple';
  if (action === 'Monitor') return 'border-brand-blue/30 bg-brand-blue/10 text-brand-blue';
  return 'border-line bg-surface-2 text-muted';
}

export function formatTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleTimeString([], { hour12: false });
}

function siemRisk(event) {
  if (event.type === 'evaluation') return { risk: 'info', tags: ['evaluation-only'] };
  if (event.type !== 'defense') return { risk: 'info', tags: ['system'] };
  const action = String(event.action || '');
  if (action === 'Remove' || action === 'Restore') return { risk: 'high', tags: ['blue-remediation'] };
  if (action === 'Analyse') return { risk: 'medium', tags: ['blue-analysis'] };
  return { risk: 'low', tags: ['blue-action'] };
}

export function buildSiemNdjson(state) {
  const index = 'cyberally-blue-audit';
  const episodeId = `cage4-seed${state.seed ?? 'unknown'}`;
  const lines = [];
  (state.events || [])
    .slice()
    .reverse()
    .forEach((event, seq) => {
      const { risk, tags } = siemRisk(event);
      const doc = {
        export_version: 'cyberally-dashboard-siem-v1',
        event_type: event.type === 'defense' ? 'blue_action' : event.type,
        episode_id: episodeId,
        seed: state.seed ?? null,
        tick: event.step ?? 0,
        timestamp: event.time,
        agent: event.agent ?? null,
        action: event.action ?? null,
        host: event.target ?? null,
        result: event.result ?? null,
        reward: event.reward ?? null,
        description: event.description,
        risk_level: risk,
        tags,
        evaluation_only: event.type === 'evaluation' || event.evaluation_only === true,
      };
      if (event.red_occupied_hosts !== undefined) doc.red_occupied_hosts = event.red_occupied_hosts;
      lines.push(JSON.stringify({ index: { _index: index, _id: `${episodeId}-${seq}` } }));
      lines.push(JSON.stringify(doc));
    });
  return lines.length ? `${lines.join('\n')}\n` : '';
}

export function classNames(...values) {
  return values.filter(Boolean).join(' ');
}
