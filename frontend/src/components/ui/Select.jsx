export default function Select({ label, className = '', children, ...props }) {
  return (
    <label className={`block text-xs text-[var(--text-muted)] ${className}`.trim()}>
      {label && <span className="mb-1.5 block font-medium text-[var(--text)]">{label}</span>}
      <select className="select-field" {...props}>
        {children}
      </select>
    </label>
  );
}
