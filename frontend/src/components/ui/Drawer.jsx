import { useEffect } from 'react';

export default function Drawer({ open, onClose, title, children, footer }) {
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

  return (
    <div className="fixed inset-0 z-[100] flex justify-end">
      <button
        type="button"
        aria-label="Close settings"
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />
      <aside
        className="relative flex h-full w-full max-w-md flex-col glass-panel border-l border-[var(--glass-border)] shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="flex items-center justify-between border-b border-[var(--border)] px-5 py-4">
          <h2 className="section-title">{title}</h2>
          <button type="button" className="btn-icon" onClick={onClose} aria-label="Close">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-5 py-4">{children}</div>
        {footer && (
          <div className="border-t border-[var(--border)] px-5 py-4">{footer}</div>
        )}
      </aside>
    </div>
  );
}
