import { AlertTriangle, CheckCircle2, Info } from 'lucide-react';

const STYLES = {
  success: 'border-brand-green/40 bg-brand-green/10 text-brand-green',
  error: 'border-brand-red/40 bg-brand-red/10 text-brand-red',
  info: 'border-brand-blue/40 bg-brand-blue/10 text-brand-blue',
};

const ICONS = {
  success: CheckCircle2,
  error: AlertTriangle,
  info: Info,
};

export default function Toasts({ toasts }) {
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-[min(24rem,calc(100vw-2rem))] flex-col gap-2" role="status" aria-live="polite">
      {toasts.map((toast) => {
        const Icon = ICONS[toast.kind] || Info;
        return (
          <div
            key={toast.id}
            className={`cage-toast pointer-events-auto flex items-start gap-2.5 rounded-xl border bg-surface px-3.5 py-3 text-sm shadow-2xl shadow-black/20 ${STYLES[toast.kind] || STYLES.info}`}
          >
            <Icon size={17} className="mt-0.5 shrink-0" />
            <span className="text-ink">{toast.message}</span>
          </div>
        );
      })}
    </div>
  );
}
