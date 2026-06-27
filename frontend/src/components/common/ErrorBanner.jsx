export default function ErrorBanner({ message, onDismiss }) {
  if (!message) return null;
  return (
    <div className="mb-4 glass-panel rounded-[var(--radius-xl)] border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
      <div className="flex items-start justify-between gap-2">
        <span>{message}</span>
        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            className="shrink-0 rounded-lg px-2 py-0.5 text-red-300 transition hover:bg-red-500/20"
          >
            Dismiss
          </button>
        )}
      </div>
    </div>
  );
}
