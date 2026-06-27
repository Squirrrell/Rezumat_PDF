export default function Slider({ label, value, onChange, min, max, step, className = '' }) {
  return (
    <label className={`block text-xs text-[var(--text-muted)] ${className}`.trim()}>
      <span className="mb-1.5 flex justify-between font-medium text-[var(--text)]">
        <span>{label}</span>
        <span className="font-mono text-[var(--accent)]">{value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </label>
  );
}
