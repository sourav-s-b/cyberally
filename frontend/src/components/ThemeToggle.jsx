import { Moon, Sun } from 'lucide-react';

export default function ThemeToggle({ theme, onToggle, className = '' }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
      className={`grid h-9 w-9 place-items-center rounded-lg border border-line bg-surface-2 text-muted transition hover:border-line-strong hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-indigo/40 ${className}`}
    >
      {theme === 'dark' ? <Moon size={17} /> : <Sun size={17} />}
    </button>
  );
}
