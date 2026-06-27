export default function LoadingSpinner({ label = 'Loading...' }) {
  return (
    <div className="flex items-center gap-2.5 text-sm font-medium text-[var(--text-muted)]">
      <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-[var(--border)] border-t-[var(--accent)]" />
      {label}
    </div>
  );
}
