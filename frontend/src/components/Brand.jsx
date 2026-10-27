import { useId } from 'react';

export default function Brand({ onClick, compact = false, className = '' }) {
  const gradientId = `shield-${useId().replace(/[:]/g, '')}`;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!onClick}
      className={`flex items-center gap-2.5 rounded-lg text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-indigo/40 ${className}`}
    >
      <svg
        viewBox="0 0 36 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className={compact ? 'h-8 w-8' : 'h-9 w-9'}
        aria-hidden="true"
      >
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#9d4edd" />
            <stop offset="50%" stopColor="#3a86ff" />
            <stop offset="100%" stopColor="#00f5d4" />
          </linearGradient>
        </defs>
        <path
          d="M18 2L33 8.5V20C33 29.5 26.5 35.5 18 38C9.5 35.5 3 29.5 3 20V8.5L18 2Z"
          stroke={`url(#${gradientId})`}
          strokeWidth="2.5"
          fill="rgba(15, 23, 42, 0.7)"
        />
        <path
          d="M18 8L27 12.5V20C27 25.5 23 29.5 18 31.5C13 29.5 9 25.5 9 20V12.5L18 8Z"
          stroke="#38bdf8"
          strokeWidth="1.5"
          strokeDasharray="2 2"
          fill="none"
        />
        <circle cx="18" cy="20" r="4" fill="#a855f7" />
      </svg>
      <span className={`font-bold tracking-tight ${compact ? 'text-base' : 'text-lg'}`}>
        Cyber<span className="text-brand-blue">Ally</span>
      </span>
    </button>
  );
}
