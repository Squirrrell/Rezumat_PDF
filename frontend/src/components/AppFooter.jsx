export default function AppFooter({ metadata, totalTime }) {
  if (!metadata) return null;

  return (
    <footer className="mt-auto border-t border-[var(--glass-border)] glass-panel">
      <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center gap-x-6 gap-y-1 px-6 py-3 text-xs text-[var(--text-muted)] lg:px-10">
        <span>
          <strong className="text-[var(--text)]">{metadata.paper_name}</strong>
        </span>
        <span>{metadata.page_count} pages</span>
        {totalTime > 0 && <span>{totalTime.toFixed(1)}s processing time</span>}
      </div>
    </footer>
  );
}
