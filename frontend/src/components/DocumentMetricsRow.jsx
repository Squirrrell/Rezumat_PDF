import Card from './ui/Card';

export default function DocumentMetricsRow({ metrics, prefix = '' }) {
  if (!metrics) return null;
  const items = [
    { label: `${prefix}Pages`, value: metrics.page_count?.toLocaleString() },
    { label: `${prefix}Words`, value: metrics.word_count?.toLocaleString() },
    { label: `${prefix}Characters`, value: metrics.character_count?.toLocaleString() },
    { label: `${prefix}Chunks`, value: metrics.chunk_count?.toLocaleString() },
    { label: `${prefix}Conversion time`, value: `${metrics.runtime_seconds?.toFixed(2)}s` },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
      {items.map((item) => (
        <Card key={item.label} interactive className="p-4">
          <div className="text-xs font-medium text-[var(--text-muted)]">{item.label}</div>
          <div className="mt-1 text-lg font-bold tracking-tight text-[var(--text)]">
            {item.value}
          </div>
        </Card>
      ))}
    </div>
  );
}
