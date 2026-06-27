import { TABS } from '../../constants';

export default function SegmentedNav({ activeId, onChange, className = '' }) {
  return (
    <nav
      className={`flex gap-1 overflow-x-auto rounded-[var(--radius-xl)] bg-[var(--surface-2)] p-1 ${className}`.trim()}
      aria-label="Main navigation"
    >
      {TABS.map((tab) => {
        const active = activeId === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onChange(tab.id)}
            className={`shrink-0 rounded-[var(--radius-lg)] px-3 py-2 text-sm font-medium transition-all duration-200 ease-in-out ${
              active
                ? 'bg-[var(--accent)] text-[var(--accent-text)] shadow-sm'
                : 'text-[var(--text-muted)] hover:bg-[var(--surface-hover)] hover:text-[var(--text)]'
            }`}
          >
            {tab.label}
          </button>
        );
      })}
    </nav>
  );
}
