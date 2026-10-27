import { Fragment } from 'react';
import {
  Activity,
  ArrowRight,
  BarChart3,
  BookOpen,
  Crosshair,
  Lock,
  Network,
  Play,
  Radar,
  Server,
  ShieldCheck,
  Star,
  Users,
} from 'lucide-react';
import Brand from './Brand.jsx';
import ThemeToggle from './ThemeToggle.jsx';
import { PrimaryButton, SecondaryButton } from '../ui.jsx';

const FEATURE_BADGES = [
  { icon: Star, tone: 'text-brand-purple', box: 'bg-brand-purple/10 border-brand-purple/25', title: ['CAGE4 Enterprise', 'Simulation'] },
  { icon: Users, tone: 'text-brand-red', box: 'bg-brand-red/10 border-brand-red/25', title: ['Native CAGE4', 'Red policies'] },
  { icon: ShieldCheck, tone: 'text-brand-blue', box: 'bg-brand-blue/10 border-brand-blue/25', title: ['Blue baseline', 'policies'] },
  { icon: Activity, tone: 'text-brand-cyan', box: 'bg-brand-cyan/10 border-brand-cyan/25', title: ['Episode events', '& metrics'] },
  { icon: Lock, tone: 'text-brand-green', box: 'bg-brand-green/10 border-brand-green/25', title: ['Local simulator', 'process'] },
];

const PIPELINE = [
  { number: '1', title: ['Native CAGE4', 'Red Agent'], icon: Crosshair, tone: 'text-brand-red', ring: 'border-brand-red/30 bg-brand-red/10', numberTone: 'bg-brand-red/15 text-brand-red', desc: 'A selected CAGE4 Red implementation acts according to its native scenario policy.' },
  { number: '2', title: ['Cyber-Range', 'Environment'], icon: Server, tone: 'text-brand-blue', ring: 'border-brand-blue/30 bg-brand-blue/10', numberTone: 'bg-brand-blue/15 text-brand-blue', desc: 'The upstream CybORG CAGE4 Enterprise simulator advances one shared world per joint step.' },
  { number: '3', title: ['CAGE4 Blue', 'Baselines'], icon: ShieldCheck, tone: 'text-brand-purple', ring: 'border-brand-purple/30 bg-brand-purple/10', numberTone: 'bg-brand-purple/15 text-brand-purple', desc: 'Existing heuristic baselines choose actions from Blue-visible observations and simulator-valid masks.' },
  { number: '4', title: ['Analytics &', 'Learning'], icon: BarChart3, tone: 'text-brand-cyan', ring: 'border-brand-cyan/30 bg-brand-cyan/10', numberTone: 'bg-brand-cyan/15 text-brand-cyan', desc: 'The dashboard records requested Blue actions, native team reward, observations and separate evaluation labels.' },
];

const KEY_FEATURES = [
  { icon: Network, tone: 'text-brand-red', box: 'bg-brand-red/10', title: 'Native CAGE4 Red policies', desc: 'Choose from the Red agent variants available in the local CAGE4 wrapper.' },
  { icon: Users, tone: 'text-brand-blue', box: 'bg-brand-blue/10', title: 'Blue baseline policies', desc: 'Compare the round-robin, masked-random and Sleep policies in seeded episodes.' },
  { icon: Activity, tone: 'text-brand-cyan', box: 'bg-brand-cyan/10', title: 'Blue-visible simulator observations', desc: 'Blue-visible CAGE4 fields only.' },
  { icon: ShieldCheck, tone: 'text-brand-green', box: 'bg-brand-green/10', title: 'Custom scenarios', desc: 'Configure vulnerable apps, networks and policies.' },
  { icon: BarChart3, tone: 'text-brand-purple', box: 'bg-brand-purple/10', title: 'Performance analytics', desc: 'Review the current CAGE4 episode event log and host evidence.' },
];

function NavLink({ href, children }) {
  return (
    <a
      href={href}
      className="rounded-md px-1 py-2 text-sm font-medium text-muted transition hover:text-ink"
    >
      {children}
    </a>
  );
}

function SimNode({ side, title, subtitle }) {
  const tone = side === 'red'
    ? 'border-brand-red/40 bg-brand-red/10 text-brand-red'
    : 'border-brand-blue/40 bg-brand-blue/10 text-brand-blue';
  return (
    <div className={`w-24 rounded-xl border px-3 py-3 text-center text-xs font-semibold backdrop-blur ${tone}`}>
      <div>{title}</div>
      <div className="mt-1 text-[10px] font-normal text-muted">{subtitle}</div>
    </div>
  );
}

export default function HomeView({ onStart, onResume, episode, onToast, theme, onToggleTheme }) {
  const hasOngoingEpisode = ['running', 'paused'].includes(episode?.status);
  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="hero-grid-bg pointer-events-none absolute inset-0" aria-hidden="true" />
      <header className="sticky top-0 z-40 border-b border-line bg-header backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-3 px-4 lg:px-8">
          <Brand onClick={() => onToast('You are on the home page.', 'info')} />
          <nav className="hidden items-center gap-7 md:flex" aria-label="Landing page">
            <a href="#home-hero" className="text-sm font-medium text-ink">Home</a>
            <NavLink href="#how-it-works-section">How It Works</NavLink>
            <NavLink href="#key-features-section">Features</NavLink>
          </nav>
          <div className="flex items-center gap-2">
            <ThemeToggle theme={theme} onToggle={onToggleTheme} />
            <SecondaryButton onClick={() => onToast('See README.md and docs/current-state.md for simulator scope and known limits.', 'info')} className="hidden sm:inline-flex">
              <BookOpen size={15} />
              Documentation
            </SecondaryButton>
          </div>
        </div>
      </header>

      <main className="relative mx-auto max-w-7xl px-4 pb-20 lg:px-8">
        <section id="home-hero" className="grid gap-12 py-16 lg:grid-cols-[1.05fr_0.95fr] lg:items-center lg:py-24">
          <div>
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-brand-blue/30 bg-brand-blue/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-brand-blue">
              <Radar size={13} />
              AI-Powered Cyber Range
            </div>
            <h1 className="text-4xl font-extrabold leading-[1.08] tracking-tight sm:text-5xl lg:text-6xl">
              Autonomous
              <br />
              Cyber-Range for
              <br />
              a <span className="gradient-text">Safer Tomorrow</span>
            </h1>
            <p className="mt-5 max-w-xl text-base leading-relaxed text-muted">
              A local CAGE4 research dashboard for running reproducible, simulation-based Red and
              Blue agent episodes and inspecting their actual observations, actions, rewards, and
              evaluation labels.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              {hasOngoingEpisode ? (
                <>
                  <PrimaryButton onClick={onResume} className="px-6 py-3">
                    <Play size={16} fill="currentColor" />
                    Resume Simulation
                  </PrimaryButton>
                  <SecondaryButton onClick={onStart} className="px-5 py-3">
                    Start New Simulation
                    <ArrowRight size={16} />
                  </SecondaryButton>
                </>
              ) : (
                <PrimaryButton onClick={onStart} className="px-6 py-3">
                  Start Simulation
                  <ArrowRight size={18} />
                </PrimaryButton>
              )}
            </div>
          </div>

          <div className="relative">
            <div className="absolute -inset-6 rounded-[2rem] bg-brand-purple/20 blur-3xl" aria-hidden="true" />
            <div className="relative overflow-hidden rounded-3xl border border-line bg-surface/80 p-6 shadow-2xl shadow-black/30 backdrop-blur sm:p-8">
              <div className="mb-6 flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-dim">
                <span>Local episode</span>
                <span className="inline-flex items-center gap-1.5 text-brand-green">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-brand-green" />
                  simulator ready
                </span>
              </div>
              <div className="flex items-center justify-between gap-2">
                <SimNode side="red" title="CAGE4 Red" subtitle="native agent" />
                <div className="h-px flex-1 bg-linear-to-r from-brand-red/40 via-brand-indigo/40 to-brand-blue/40" />
                <div className="flex flex-col items-center text-center">
                  <div className="grid h-24 w-24 place-items-center rounded-full border border-brand-purple/40 bg-surface-2 shadow-[0_0_40px_rgba(168,85,247,0.35)]">
                    <ShieldCheck size={34} className="text-brand-purple" />
                  </div>
                  <div className="mt-3 text-sm font-bold">CybORG</div>
                  <div className="text-[10px] text-muted">Enterprise scenario</div>
                  <div className="text-[10px] text-dim">local simulator</div>
                </div>
                <div className="h-px flex-1 bg-linear-to-r from-brand-blue/40 via-brand-indigo/40 to-brand-purple/40" />
                <SimNode side="blue" title="CAGE4 Blue" subtitle="baseline policy" />
              </div>
              <div className="mt-6 grid grid-cols-3 gap-2 text-center">
                {[
                  ['Shared world', 'one joint step'],
                  ['Blue view only', 'policy inputs'],
                  ['Evaluation', 'privileged labels'],
                ].map(([label, detail]) => (
                  <div key={label} className="rounded-lg border border-line bg-surface-2 px-2 py-2">
                    <div className="text-[11px] font-semibold text-ink">{label}</div>
                    <div className="text-[10px] text-dim">{detail}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="pb-4">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-5">
            {FEATURE_BADGES.map(({ icon: Icon, tone, box, title }) => (
              <div key={title.join(' ')} className="flex items-center gap-3 rounded-xl border border-line bg-surface/70 px-3 py-3">
                <div className={`grid h-10 w-10 shrink-0 place-items-center rounded-lg border ${box} ${tone}`}>
                  <Icon size={19} />
                </div>
                <div className="text-xs font-semibold leading-tight">
                  {title[0]}
                  <br />
                  {title[1]}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section id="how-it-works-section" className="py-16 lg:py-20">
          <div className="mb-10 max-w-2xl">
            <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">How It Works</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              A seeded CAGE4 Enterprise episode runs Red and Blue agents in one shared simulator.
            </p>
          </div>
          <div className="grid gap-4 lg:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr] lg:items-center">
            {PIPELINE.map(({ number, title, icon: Icon, tone, ring, numberTone, desc }, index) => (
              <Fragment key={number}>
                <div className="relative rounded-2xl border border-line bg-surface p-5">
                  <div className={`absolute -top-3 left-5 grid h-7 w-7 place-items-center rounded-full text-xs font-bold ${numberTone}`}>
                    {number}
                  </div>
                  <div className={`mb-4 mt-2 grid h-11 w-11 place-items-center rounded-xl border ${ring} ${tone}`}>
                    <Icon size={21} />
                  </div>
                  <h3 className={`text-sm font-bold leading-tight ${tone}`}>
                    {title[0]}
                    <br />
                    {title[1]}
                  </h3>
                  <p className="mt-2 text-xs leading-relaxed text-muted">{desc}</p>
                </div>
                {index < PIPELINE.length - 1 ? (
                  <div className="hidden justify-center lg:flex" aria-hidden="true">
                    <ArrowRight size={20} className="text-brand-blue/70" />
                  </div>
                ) : null}
              </Fragment>
            ))}
          </div>
        </section>

        <section id="key-features-section" className="py-10 lg:py-14">
          <div className="mb-10 max-w-2xl">
            <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">Key Features</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Inspect the simulator behavior currently implemented in this repository.
            </p>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {KEY_FEATURES.map(({ icon: Icon, tone, box, title, desc }) => (
              <div key={title} className="rounded-2xl border border-line bg-surface p-5 transition hover:border-line-strong hover:bg-surface-hover">
                <div className={`mb-4 grid h-11 w-11 place-items-center rounded-xl ${box} ${tone}`}>
                  <Icon size={21} />
                </div>
                <h3 className="text-sm font-bold">{title}</h3>
                <p className="mt-2 text-xs leading-relaxed text-muted">{desc}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="pb-4 pt-10">
          <div className="relative overflow-hidden rounded-3xl border border-brand-indigo/30 bg-linear-to-br from-brand-indigo/20 via-surface to-brand-purple/15 p-8 sm:p-12">
            <div className="absolute -right-10 -top-10 h-48 w-48 rounded-full bg-brand-purple/20 blur-3xl" aria-hidden="true" />
            <div className="relative">
              <div className="mb-3 inline-flex items-center gap-2 text-brand-purple">
                <Radar size={18} />
                <span className="text-xs font-semibold uppercase tracking-[0.18em]">Local CAGE4</span>
              </div>
              <h3 className="max-w-2xl text-2xl font-bold tracking-tight sm:text-3xl">
                Ready to experience an autonomous cyber battle?
              </h3>
              <p className="mt-3 max-w-xl text-sm leading-relaxed text-muted">
                Launch the simulation environment and explore real-time attacks and adaptive
                defenses.
              </p>
              <PrimaryButton onClick={onStart} className="mt-6 px-6 py-3">
                Start Simulation
                <ArrowRight size={18} />
              </PrimaryButton>
            </div>
          </div>
        </section>
      </main>

      <footer className="relative border-t border-line px-4 py-8 lg:px-8">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 text-xs text-dim sm:flex-row sm:items-center sm:justify-between">
          <span>CyberAlly · local CAGE4 research dashboard</span>
          <span>Simulator view only — not a container range or production SIEM.</span>
        </div>
      </footer>
    </div>
  );
}
