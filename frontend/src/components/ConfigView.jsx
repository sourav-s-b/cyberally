import { useEffect, useState } from 'react';
import { AlertTriangle, ArrowLeft, ArrowRight, Database, Info, Loader2 } from 'lucide-react';
import Brand from './Brand.jsx';
import ThemeToggle from './ThemeToggle.jsx';
import { FieldLabel, Panel, PanelNote, PrimaryButton, SecondaryButton, inputClass, selectClass } from '../ui.jsx';
import { humanizeCheckpoint } from '../format.js';

const POLICY_GROUPS = [
  { value: 'baseline', label: 'Baselines' },
  { value: 'shortlist', label: '7 candidate models' },
  { value: 'mappo', label: 'MAPPO experiments' },
  { value: 'bc', label: 'Behavior cloning' },
  { value: 'iql', label: 'IQL / offline AWR' },
  { value: 'rvs', label: 'Return-conditioned RvS' },
  { value: 'setactor', label: 'SetActor experiments' },
  { value: 'harness', label: 'Main32 harness PPO' },
  { value: 'hybrid', label: 'SetActor / attention hybrids' },
];

const SHORTLIST_ORDER = [
  'architecture:fixed_s0',
  'architecture:hybrid_s0',
  'architecture:set_s0',
  'harness:gpt_harness/main32_status/s0/pilot/both_s0',
  'bc_fact_attn1_20261002/0',
  'rvs_14/0',
  'iql_14a05/0',
];

const BASELINE_SUMMARY = {
  round_robin: 'Recommended starting point: predictable Lancer-style host coverage.',
  sleep: 'Negative control: Blue always chooses Sleep.',
  masked_random: 'Control policy: chooses randomly from simulator-valid actions.',
};

export default function ConfigView({
  models,
  modelsError,
  modelsNotice,
  config,
  updateConfig,
  onLaunch,
  onBack,
  onHome,
  onToast,
  theme,
  onToggleTheme,
}) {
  const [launching, setLaunching] = useState(false);

  const family = config.policyKind;
  const isBaseline = family === 'baseline';
  const matches = models
    .filter((model) => (family === 'shortlist' ? model.shortlist : model.family === family))
    .sort((a, b) => (family === 'shortlist'
      ? SHORTLIST_ORDER.indexOf(a.id) - SHORTLIST_ORDER.indexOf(b.id)
      : 0))
    .filter((model) => model.compatible);
  const compatibleCount = models.filter((model) => model.compatible).length;
  const experimentalFamilies = POLICY_GROUPS.filter(
    (group) => group.value !== 'baseline' && group.value !== 'shortlist',
  ).length;

  const firstMatchId = matches[0]?.id || '';
  const hasCurrentMatch = matches.some((model) => model.id === config.modelId);

  useEffect(() => {
    if (isBaseline || hasCurrentMatch) return;
    updateConfig({ modelId: firstMatchId });
  }, [isBaseline, hasCurrentMatch, firstMatchId, updateConfig]);

  const selectedModel = models.find((model) => model.id === config.modelId);
  const bluePolicy = isBaseline ? config.baseline : config.modelId;

  let summary;
  if (isBaseline) {
    summary = BASELINE_SUMMARY[config.baseline];
  } else if (selectedModel) {
    const inputs = selectedModel.temporal_features?.length
      ? selectedModel.temporal_features.join(', ')
      : 'base CAGE4 observations';
    summary = `${selectedModel.family_label} · ${selectedModel.architecture}. Inputs: ${inputs}${selectedModel.return_conditioned ? ' · return-conditioned (target −80)' : ''}.${selectedModel.kind === 'checkpoint' ? ` ${selectedModel.status}` : ''}`;
  } else {
    summary = modelsNotice
      ? 'Saved actor checkpoints are unavailable in this server environment.'
      : `No compatible ${family.toUpperCase()} actor is installed locally. Pick another group or a baseline.`;
  }

  const launch = async () => {
    if (!bluePolicy) {
      onToast('No compatible actor is selected. Choose another policy group or a baseline.', 'error');
      return;
    }
    setLaunching(true);
    try {
      await onLaunch({
        seed: Number(config.seed),
        steps: Number(config.horizon),
        red_agent: config.redAgent,
        red_model: config.redModel,
        blue_policy: bluePolicy,
      });
    } finally {
      setLaunching(false);
    }
  };

  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-header backdrop-blur">
        <div className="mx-auto flex h-16 max-w-4xl items-center justify-between gap-3 px-4">
          <Brand onClick={onHome} />
          <div className="flex items-center gap-2">
            <ThemeToggle theme={theme} onToggle={onToggleTheme} />
            <SecondaryButton onClick={onBack} className="px-3">
              <ArrowLeft size={15} />
              <span className="hidden sm:inline">Back to Home</span>
            </SecondaryButton>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-4 py-10">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Configure CAGE4 Episode</h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
          Pick a Blue policy from the fixed drop-down list, choose a native CAGE4 Red agent, then
          launch a seeded episode in the local Enterprise simulator.
        </p>

        <Panel className="mt-8 overflow-hidden">
          <div className="border-b border-line bg-surface-2/60 px-5 py-4 sm:px-6">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 className="text-sm font-semibold">Blue policy</h2>
                <p className="mt-1 text-xs text-muted">
                  Start with the dependable baseline, or browse an experimental checkpoint.
                </p>
              </div>
              <div className="w-full sm:w-56">
                <FieldLabel htmlFor="policy-kind">Policy group</FieldLabel>
                <select
                  id="policy-kind"
                  className={selectClass}
                  value={family}
                  onChange={(event) => updateConfig({ policyKind: event.target.value })}
                >
                  {POLICY_GROUPS.map((group) => (
                    <option key={group.value} value={group.value}>{group.label}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <div className="space-y-4 px-5 py-5 sm:px-6">
            {isBaseline ? (
              <div>
                <FieldLabel htmlFor="blue-baseline">Blue baseline</FieldLabel>
                <select
                  id="blue-baseline"
                  className={selectClass}
                  value={config.baseline}
                  onChange={(event) => updateConfig({ baseline: event.target.value })}
                >
                  <option value="round_robin">Round-robin · Lancer-style coverage</option>
                  <option value="sleep">Sleep · no actions (negative control)</option>
                  <option value="masked_random">Masked random · legal-action control</option>
                </select>
              </div>
            ) : (
              <div>
                <FieldLabel htmlFor="blue-model">Saved actor / checkpoint</FieldLabel>
                <select
                  id="blue-model"
                  className={selectClass}
                  value={config.modelId}
                  onChange={(event) => updateConfig({ modelId: event.target.value })}
                  disabled={!matches.length}
                >
                  {matches.length ? matches.map((model) => (
                    <option key={model.id} value={model.id}>
                      {humanizeCheckpoint(model)} — {model.architecture}
                    </option>
                  )) : (
                    <option value="">
                      {modelsNotice || modelsError
                        ? 'Model catalog unavailable — baselines still work'
                        : `No compatible ${family.toUpperCase()} actor installed`}
                    </option>
                  )}
                </select>
                <p className="mt-2 text-xs leading-relaxed text-dim">
                  {family === 'shortlist'
                    ? 'The seven documented comparison candidates, including locally retrained gate-ablation models. Every raw checkpoint remains available in its family group.'
                    : 'Fixed list of locally saved actors that match the CAGE4 observation and action layout. No checkpoint path typing required.'}
                </p>
                {modelsNotice ? (
                  <div className="mt-3 flex items-start gap-2 rounded-xl border border-brand-blue/25 bg-brand-blue/10 px-3 py-2 text-[11px] leading-relaxed text-brand-blue">
                    <Info size={13} className="mt-0.5 shrink-0" />
                    <span>{modelsNotice}</span>
                  </div>
                ) : null}
              </div>
            )}

            <div className={`flex items-start gap-2 rounded-xl border px-4 py-3 text-xs leading-relaxed ${
              modelsError
                ? 'border-brand-red/30 bg-brand-red/10 text-brand-red'
                : 'border-line bg-surface-2 text-muted'
            }`}
            >
              {modelsError ? <AlertTriangle size={14} className="mt-0.5 shrink-0" /> : null}
              <p>{modelsError || summary}</p>
            </div>

            {selectedModel && selectedModel.kind !== 'checkpoint' ? (
              <div className="rounded-xl border border-brand-indigo/25 bg-brand-indigo/5 px-4 py-3 text-xs leading-relaxed text-muted">
                <p className="font-semibold text-ink">
                  {selectedModel.kind === 'set_actor'
                    ? 'SetActor pilot artifact'
                    : selectedModel.kind === 'harness_residual'
                      ? 'Main32 harness artifact'
                      : 'Gate-ablation artifact'}
                </p>
                <p className="mt-1">{selectedModel.status}</p>
                <dl className="mt-3 space-y-1 font-mono text-[10px] leading-relaxed text-dim">
                  <div className="break-all">
                    <dt className="inline font-sans font-semibold text-muted">Actor SHA-256 </dt>
                    <dd className="inline">{selectedModel.actor_sha256}</dd>
                  </div>
                  <div className="break-all">
                    <dt className="inline font-sans font-semibold text-muted">Scorer SHA-256 </dt>
                    <dd className="inline">{selectedModel.scorer_sha256}</dd>
                  </div>
                  <div>
                    <dt className="inline font-sans font-semibold text-muted">Manifest </dt>
                    <dd className="inline">{selectedModel.manifest_version || 'gate-ablation'} · {selectedModel.manifest_status}</dd>
                  </div>
                </dl>
              </div>
            ) : null}

            <p className="flex items-center gap-2 text-xs text-dim">
              <Database size={14} />
              {compatibleCount} local checkpoint{compatibleCount === 1 ? '' : 's'} available across {experimentalFamilies} experimental families.
            </p>
          </div>
        </Panel>

        <Panel className="mt-4 p-5 sm:p-6">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <div>
              <FieldLabel htmlFor="red-agent">Red opponent</FieldLabel>
              <select
                id="red-agent"
                className={selectClass}
                value={config.redAgent}
                onChange={(event) => updateConfig({ redAgent: event.target.value })}
              >
                <option value="discovery">DiscoveryFSRed (default)</option>
                <option value="finite">FiniteStateRedAgent</option>
                <option value="verbose">VerboseFSRed</option>
                <option value="random">RandomSelectRedAgent</option>
                <option value="sleep">SleepAgent</option>
                <option value="llm-planner">LLM planner · local Ollama</option>
              </select>
            </div>
            {config.redAgent === 'llm-planner' ? (
              <div>
                <FieldLabel htmlFor="red-model">Ollama model tag</FieldLabel>
                <input
                  id="red-model"
                  type="text"
                  className={inputClass}
                  value={config.redModel}
                  onChange={(event) => updateConfig({ redModel: event.target.value })}
                  placeholder="qwen2.5:3b"
                />
              </div>
            ) : null}
            <div>
              <FieldLabel htmlFor="seed">Seed</FieldLabel>
              <input
                id="seed"
                type="number"
                min="0"
                className={inputClass}
                value={config.seed}
                onChange={(event) => updateConfig({ seed: event.target.value })}
              />
            </div>
            <div>
              <FieldLabel htmlFor="horizon">Horizon</FieldLabel>
              <input
                id="horizon"
                type="number"
                min="3"
                max="400"
                className={inputClass}
                value={config.horizon}
                onChange={(event) => updateConfig({ horizon: event.target.value })}
              />
            </div>
          </div>
          {config.redAgent === 'llm-planner' ? (
            <p className="mt-4 text-xs leading-relaxed text-dim">
              LLM Red runs against local Ollama at 127.0.0.1:11434. The dashboard caps
              planning at 12 requests and two plans per Red agent, so short horizons
              show the planner/action trace without a long inference run.
            </p>
          ) : null}
        </Panel>

        <div className="mt-6 flex justify-end">
          <PrimaryButton onClick={launch} disabled={launching || !bluePolicy} className="px-6 py-3">
            {launching ? <Loader2 size={17} className="animate-spin" /> : <ArrowRight size={17} />}
            {launching ? 'Launching…' : 'Launch CAGE4 episode'}
          </PrimaryButton>
        </div>

        <PanelNote className="mt-6">
          <strong className="text-ink">Scope:</strong> the dashboard launches the local CybORG
          CAGE4 Enterprise simulator. It does not control containers, real hosts or network
          telemetry, and privileged labels never enter Blue policy inputs.
        </PanelNote>
      </main>
    </div>
  );
}
