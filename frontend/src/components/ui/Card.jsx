export default function Card({ children, className = '', interactive = false, as: Tag = 'div', ...props }) {
  const classes = [
    'glass-panel rounded-[var(--radius-xl)] p-5',
    interactive ? 'card-interactive cursor-default' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <Tag className={classes} {...props}>
      {children}
    </Tag>
  );
}
