export const selectClass =
  'w-full rounded-lg border border-line bg-surface-2 px-3 py-2 text-sm text-ink outline-none transition focus:border-line-strong focus:ring-2 focus:ring-brand-indigo/30 disabled:cursor-not-allowed disabled:opacity-50';

export const inputClass = selectClass;

export function Panel({ className = '', children }) {
  return (
    <section className={`rounded-2xl border border-line bg-surface shadow-xl shadow-black/10 ${className}`}>
      {children}
    </section>
  );
}

export function PanelHeader({ title, subtitle, actions, className = '' }) {
  return (
    <div className={`flex flex-wrap items-start justify-between gap-3 border-b border-line px-5 py-4 ${className}`}>
      <div className="min-w-0">
        <h3 className="text-sm font-semibold tracking-tight text-ink">{title}</h3>
        {subtitle ? <p className="mt-1 max-w-2xl text-xs leading-relaxed text-muted">{subtitle}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  );
}

export function FieldLabel({ children, htmlFor }) {
  return (
    <label htmlFor={htmlFor} className="mb-1.5 block text-xs font-semibold uppercase tracking-wider text-dim">
      {children}
    </label>
  );
}

const buttonBase =
  'inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-indigo/40 disabled:cursor-not-allowed disabled:opacity-50';

export function PrimaryButton({ className = '', children, ...props }) {
  return (
    <button
      type="button"
      className={`${buttonBase} bg-linear-to-r from-brand-indigo via-brand-blue to-brand-indigo text-white shadow-lg shadow-brand-indigo/25 hover:brightness-110 ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function SecondaryButton({ className = '', children, ...props }) {
  return (
    <button
      type="button"
      className={`${buttonBase} border border-line bg-surface-2 text-ink hover:border-line-strong hover:bg-surface-hover ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function GhostButton({ className = '', children, ...props }) {
  return (
    <button
      type="button"
      className={`${buttonBase} text-muted hover:bg-surface-hover hover:text-ink ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function Badge({ children, className = '' }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-semibold ${className}`}>
      {children}
    </span>
  );
}

export function PanelNote({ children, className = '' }) {
  return (
    <p className={`rounded-xl border border-line bg-surface-2 px-4 py-3 text-xs leading-relaxed text-muted ${className}`}>
      {children}
    </p>
  );
}
