export default function Button({
  children,
  variant = 'primary',
  className = '',
  type = 'button',
  ...props
}) {
  const base = variant === 'primary' ? 'btn-primary' : 'btn-ghost';
  return (
    <button type={type} className={`${base} ${className}`.trim()} {...props}>
      {children}
    </button>
  );
}
