import { useEffect, useState } from 'react';
import { fetchHealth } from '../api/client';
import { APP_NAME, APP_TAGLINE } from '../constants';
import { THEMES, useTheme } from '../context/ThemeContext';
import SegmentedNav from './ui/SegmentedNav';

export default function AppHeader({
  activeTab,
  onTabChange,
  onOpenSettings,
  onOpenRuntime,
}) {
  const { theme, setTheme } = useTheme();
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const statusOk = health?.status === 'ok';
  const cudaOk = health?.cuda_available;

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-[var(--glass-border)]">
      <div className="mx-auto flex h-[var(--header-height)] max-w-6xl items-center gap-3 px-4 lg:px-10">
        <div className="flex min-w-0 shrink-0 items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-[var(--radius-lg)] bg-[var(--accent)] text-xs font-bold text-[var(--accent-text)] shadow-sm">
            SQ
          </div>
          <div className="hidden sm:block">
            <p className="text-sm font-bold tracking-tight text-[var(--text)]">{APP_NAME}</p>
            <p className="text-[10px] text-[var(--text-muted)]">{APP_TAGLINE}</p>
          </div>
        </div>

        <div className="hidden min-w-0 flex-1 justify-center md:flex">
          <SegmentedNav activeId={activeTab} onChange={onTabChange} className="max-w-full" />
        </div>

        <div className="ml-auto flex shrink-0 items-center gap-1">
          <div className="hidden items-center gap-1.5 md:flex">
            {THEMES.map((t) => (
              <button
                key={t.id}
                type="button"
                title={t.label}
                onClick={() => setTheme(t.id)}
                className={`h-7 w-7 rounded-full border-2 transition-all duration-200 ${
                  theme === t.id ? 'scale-110' : 'opacity-70 hover:scale-105 hover:opacity-100'
                }`}
                style={{
                  background: t.swatch,
                  borderColor: theme === t.id ? t.ring : 'var(--border)',
                }}
              />
            ))}
          </div>

          <button
            type="button"
            className="btn-icon"
            onClick={onOpenRuntime}
            title={statusOk ? (cudaOk ? 'Backend online (GPU)' : 'Backend online (CPU)') : 'Backend offline'}
            aria-label="System status"
          >
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                statusOk ? (cudaOk ? 'bg-[var(--success)]' : 'bg-[var(--warning)]') : 'bg-[var(--danger)]'
              }`}
              style={{ boxShadow: statusOk ? `0 0 8px ${cudaOk ? 'var(--success)' : 'var(--warning)'}` : undefined }}
            />
          </button>

          <button
            type="button"
            className="btn-icon"
            onClick={onOpenSettings}
            aria-label="Open settings"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
              />
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>
      </div>

      <div className="border-t border-[var(--border)] px-4 py-2 md:hidden">
        <SegmentedNav activeId={activeTab} onChange={onTabChange} />
      </div>
    </header>
  );
}
