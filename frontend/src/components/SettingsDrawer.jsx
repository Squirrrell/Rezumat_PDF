import { useEffect, useState } from 'react';
import { fetchHealth } from '../api/client';
import { THEMES, useTheme } from '../context/ThemeContext';
import { useSettings } from '../hooks/useSettings';
import Select from './ui/Select';
import Slider from './ui/Slider';

function SettingsSection({ title, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-[var(--border)] pb-4 last:border-0">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between py-2 text-left text-sm font-semibold text-[var(--text)] transition hover:text-[var(--accent)]"
      >
        {title}
        <svg
          className={`h-4 w-4 text-[var(--text-muted)] transition-transform ${open ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {open && <div className="space-y-4 pt-2">{children}</div>}
    </div>
  );
}

export default function SettingsDrawer({ open, onClose, scrollToRuntime = false }) {
  const s = useSettings();
  const { theme, setTheme } = useTheme();
  const [health, setHealth] = useState(null);

  useEffect(() => {
    if (!open) return;
    fetchHealth().then(setHealth).catch(() => setHealth(null));
  }, [open]);

  useEffect(() => {
    if (!open || !scrollToRuntime) return;
    const el = document.getElementById('settings-runtime');
    el?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [open, scrollToRuntime]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
    };
  }, [open, onClose]);

  if (!open) return null;

  const themeFooter = (
    <div>
      <p className="mb-2 text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
        Theme
      </p>
      <div className="flex gap-2">
        {THEMES.map((t) => (
          <button
            key={t.id}
            type="button"
            title={t.label}
            onClick={() => setTheme(t.id)}
            className={`h-8 w-8 rounded-full border-2 transition ${
              theme === t.id ? 'scale-110' : 'opacity-80 hover:scale-105'
            }`}
            style={{
              background: t.swatch,
              borderColor: theme === t.id ? t.ring : 'var(--border)',
            }}
          />
        ))}
      </div>
    </div>
  );

  return (
    <>
      <button
        type="button"
        aria-label="Close settings"
        className="fixed inset-0 z-[100] bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />
      <aside
        className="fixed top-0 right-0 z-[101] flex h-full w-full max-w-md flex-col glass-panel border-l border-[var(--glass-border)] shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
      >
        <div className="flex items-center justify-between border-b border-[var(--border)] px-5 py-4">
          <h2 className="section-title">Settings</h2>
          <button type="button" className="btn-icon" onClick={onClose} aria-label="Close">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
          <SettingsSection title="Indexing">
            <Slider
              label="Chunk size (characters)"
              value={s.chunkSize}
              onChange={s.setChunkSize}
              min={800}
              max={1500}
              step={50}
            />
            <Slider
              label="Chunk overlap (characters)"
              value={s.overlap}
              onChange={s.setOverlap}
              min={50}
              max={300}
              step={10}
            />
          </SettingsSection>

          <SettingsSection title="Summarization" defaultOpen={false}>
            <Select
              label="Summary length"
              value={s.summaryLength}
              onChange={(e) => s.setSummaryLength(e.target.value)}
            >
              <option value="short">short</option>
              <option value="medium">medium</option>
              <option value="long">long</option>
            </Select>
            <Slider
              label="Max chunks"
              value={s.summaryMaxChunks}
              onChange={s.setSummaryMaxChunks}
              min={3}
              max={20}
              step={1}
            />
          </SettingsSection>

          <SettingsSection title="Q&A & test cards">
            <Select
              label="Instruct model"
              value={s.instructModelChoice}
              onChange={(e) => s.setInstructModelChoice(e.target.value)}
            >
              <option value="Qwen2.5-0.5B-Instruct">Qwen2.5-0.5B-Instruct</option>
              <option value="flan-t5-small (fallback)">flan-t5-small (fallback)</option>
            </Select>
            <Select
              label="Answer length"
              value={s.answerLength}
              onChange={(e) => s.setAnswerLength(e.target.value)}
            >
              <option value="short">short</option>
              <option value="medium">medium</option>
              <option value="long">long</option>
            </Select>
            <Slider
              label="Retrieved chunks for Q&A"
              value={s.qaTopK}
              onChange={s.setQaTopK}
              min={1}
              max={10}
              step={1}
            />
          </SettingsSection>

          <SettingsSection title="Runtime & models">
            <div id="settings-runtime" className="space-y-2 text-xs text-[var(--text-muted)]">
              {health ? (
                <>
                  <p>
                    <strong className="text-[var(--text)]">Device:</strong> {health.device}
                    {health.cuda_available && health.gpu_name
                      ? ` (${health.gpu_name})`
                      : !health.cuda_available
                        ? ' (CPU — install CUDA PyTorch for GPU)'
                        : ''}
                  </p>
                  <p>
                    <strong className="text-[var(--text)]">CUDA:</strong>{' '}
                    {health.cuda_available ? 'available' : 'not available'}
                  </p>
                  <p>
                    <strong className="text-[var(--text)]">Marker device:</strong>{' '}
                    {health.marker_device}
                  </p>
                  <p className="break-all">
                    <strong className="text-[var(--text)]">Converter:</strong>{' '}
                    {health.converter_model}
                  </p>
                  <p className="break-all">
                    <strong className="text-[var(--text)]">Embeddings:</strong>{' '}
                    {health.embedding_model}
                  </p>
                  <p className="break-all">
                    <strong className="text-[var(--text)]">Qwen:</strong> {health.qwen_model}
                  </p>
                  <p className="break-all">
                    <strong className="text-[var(--text)]">FLAN:</strong> {health.flan_model}
                  </p>
                </>
              ) : (
                <p>Backend not reachable.</p>
              )}
            </div>
          </SettingsSection>
        </div>

        <div className="border-t border-[var(--border)] px-5 py-4 md:hidden">{themeFooter}</div>
      </aside>
    </>
  );
}
