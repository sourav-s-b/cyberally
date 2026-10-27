import { useEffect, useState } from 'react';
import {
  AlertTriangle,
  Archive,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Crown,
  Dices,
  Info,
  Landmark,
  Loader2,
  Moon,
  Pyramid,
  Radar,
  Repeat2,
  TreePine,
} from 'lucide-react';
import Brand from './Brand.jsx';
import ThemeToggle from './ThemeToggle.jsx';
import { FieldLabel, Panel, PrimaryButton, SecondaryButton, inputClass, selectClass } from '../ui.jsx';
import { humanizeCheckpoint } from '../format.js';

const POLICY_GROUPS = [
  { value: 'baseline', label: 'Baselines' },
  { value: 'shortlist', label: 'Seven candidates' },
  { value: 'mappo', label: 'MAPPO experiments' },
  { value: 'bc', label: 'Behavior cloning' },
  { value: 'iql', label: 'IQL / offline AWR' },
  { value: 'rvs', label: 'Return-conditioned RvS' },
  { value: 'setactor', label: 'SetActor experiments' },
  { value: 'harness', label: 'Main32 harness PPO' },
  { value: 'hybrid', label: 'SetActor / attention hybrids' },
];

const BASELINE_CARDS = [
  {
    value: 'round_robin',
    name: 'Round-robin',
    detail: 'Lancer-style coverage',
    icon: Repeat2,
    tone: 'bg-brand-blue/10 text-brand-blue',
  },
  {
    value: 'sleep',
    name: 'Sleep',
    detail: 'Negative control',
    icon: Moon,
    tone: 'bg-brand-indigo/10 text-brand-indigo',
  },
  {
    value: 'masked_random',
    name: 'Masked random',
    detail: 'Legal-action control',
    icon: Dices,
    tone: 'bg-brand-purple/10 text-brand-purple',
  },
];

const SHORTLIST_CARDS = [
  {
    id: 'architecture:fixed_s0',
    name: 'Zeus',
    detail: 'Fixed 0.5 SetActor / attention blend',
    icon: Landmark,
    tone: 'bg-brand-orange/10 text-brand-orange',
  },
  {
    id: 'architecture:hybrid_s0',
    name: 'Hera',
    detail: 'Learned SetActor / attention gate',
    icon: TreePine,
    tone: 'bg-brand-green/10 text-brand-green',
  },
  {
    id: 'architecture:set_s0',
    name: 'Poseidon',
    detail: 'SetActor',
    icon: Pyramid,
    tone: 'bg-brand-yellow/10 text-brand-yellow',
  },
  {
    id: 'harness:gpt_harness/main32_status/s0/pilot/both_s0',
    name: 'Hades',
    detail: 'Main32 harness PPO',
    icon: Radar,
    tone: 'bg-brand-cyan/10 text-brand-cyan',
  },
  {
    id: 'bc_fact_attn1_20261002/0',
    name: 'Athena',
    detail: 'Behavior cloning',
    icon: BookOpen,
    tone: 'bg-brand-indigo/10 text-brand-indigo',
  },
  {
    id: 'rvs_14/0',
    name: 'Apollo',
    detail: 'Return-conditioned RvS',
    icon: Crown,
    tone: 'bg-brand-purple/10 text-brand-purple',
  },
  {
    id: 'iql_14a05/0',
    name: 'Artemis',
    detail: 'IQL / offline AWR',
    icon: Archive,
    tone: 'bg-brand-red/10 text-brand-red',
  },
];

const SHORTLIST_ORDER = SHORTLIST_CARDS.map((card) => card.id);

function ChoiceCard({ selected, disabled, icon: Icon, name, detail, tone, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={selected}
      className={`group flex min-h-24 items-start gap-3 rounded-xl border p-3 text-left transition ${
        selected
          ? 'border-brand-indigo/60 bg-brand-indigo/10 shadow-lg shadow-brand-indigo/10'
          : 'border-line bg-surface-2 hover:border-line-strong hover:bg-surface-hover'
      } ${disabled ? 'cursor-not-allowed opacity-45' : ''}`}
    >
      <span className={`grid h-10 w-10 shrink-0 place-items-center rounded-lg ${tone}`}>
        <Icon size={19} />
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-semibold leading-tight text-ink">{name}</span>
        <span className="mt-1 block text-[11px] leading-snug text-muted">{detail}</span>
        {selected ? (
          <span className="mt-2 block text-[10px] font-semibold uppercase tracking-wider text-brand-blue">
            Selected
          </span>
        ) : null}
      </span>
    </button>
  );
}

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
  const isShortlist = family === 'shortlist';
  const matches = models
    .filter((model) => (isShortlist ? model.shortlist : model.family === family))
    .sort((a, b) => (isShortlist
      ? SHORTLIST_ORDER.indexOf(a.id) - SHORTLIST_ORDER.indexOf(b.id)
      : 0))
    .filter((model) => model.compatible);

  const firstMatchId = matches[0]?.id || '';
  const hasCurrentMatch = matches.some((model) => model.id === config.modelId);

  useEffect(() => {
    if (isBaseline || hasCurrentMatch) return;
    updateConfig({ modelId: firstMatchId });
  }, [isBaseline, hasCurrentMatch, firstMatchId, updateConfig]);

  const bluePolicy = isBaseline ? config.baseline : config.modelId;
  const statusNote = modelsError || modelsNotice;

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
        <div className="mx-auto flex h-16 max-w-5xl items-center justify-between gap-3 px-4">
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

      <main className="mx-auto max-w-5xl px-4 py-10">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Configure CAGE4 Episode</h1>
        <p className="mt-2 text-sm text-muted">
          Choose a Blue policy and a native Red agent, then launch a seeded CAGE4 episode.
        </p>

        <Panel className="mt-8 overflow-hidden">
          <div className="border-b border-line bg-surface-2/60 px-5 py-4 sm:px-6">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 className="text-sm font-semibold">Blue policy</h2>
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
                <FieldLabel>Blue baseline</FieldLabel>
                <div className="grid gap-3 sm:grid-cols-3">
                  {BASELINE_CARDS.map((card) => (
                    <ChoiceCard
                      key={card.value}
                      {...card}
                      selected={config.baseline === card.value}
                      onClick={() => updateConfig({ baseline: card.value })}
                    />
                  ))}
                </div>
              </div>
            ) : isShortlist ? (
              <div>
                <FieldLabel>Candidate model</FieldLabel>
                <div className="grid gap-3 sm:grid-cols-2">
                  {SHORTLIST_CARDS.map((card) => {
                    const model = models.find((item) => item.id === card.id);
                    return (
                      <ChoiceCard
                        key={card.id}
                        {...card}
                        selected={config.modelId === card.id}
                        disabled={!model?.compatible}
                        onClick={() => updateConfig({ modelId: card.id })}
                      />
                    );
                  })}
                </div>
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
              </div>
            )}

            {statusNote ? (
              <div className={`flex items-start gap-2 rounded-xl border px-4 py-3 text-xs leading-relaxed ${
                modelsError
                  ? 'border-brand-red/30 bg-brand-red/10 text-brand-red'
                  : 'border-brand-blue/25 bg-brand-blue/10 text-brand-blue'
              }`}
              >
                {modelsError ? <AlertTriangle size={14} className="mt-0.5 shrink-0" /> : <Info size={14} className="mt-0.5 shrink-0" />}
                <p>{statusNote}</p>
              </div>
            ) : null}
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
                max="1500"
                className={inputClass}
                value={config.horizon}
                onChange={(event) => updateConfig({ horizon: event.target.value })}
              />
            </div>
          </div>
          {config.redAgent === 'llm-planner' ? (
            <p className="mt-3 text-xs text-dim">
              Local Ollama. Demo cap: 12 requests and two plans per Red agent.
            </p>
          ) : null}
        </Panel>

        <div className="mt-6 flex justify-end">
          <PrimaryButton onClick={launch} disabled={launching || !bluePolicy} className="px-6 py-3">
            {launching ? <Loader2 size={17} className="animate-spin" /> : <ArrowRight size={17} />}
            {launching ? 'Launching…' : 'Launch CAGE4 episode'}
          </PrimaryButton>
        </div>

      </main>
    </div>
  );
}
