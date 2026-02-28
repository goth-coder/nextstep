interface IndicatorCardProps {
  label: string
  value: number | null
}

export default function IndicatorCard({ label, value }: IndicatorCardProps) {
  const isAvailable = value !== null && value !== undefined

  return (
    <div
      style={{
        padding: '0.875rem 1rem',
        background: '#fff',
        border: '1px solid #e5e7eb',
        borderRadius: '0.5rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.25rem',
      }}
    >
      <span
        style={{
          fontSize: '0.75rem',
          fontWeight: 600,
          color: '#6b7280',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        }}
      >
        {label}
      </span>
      {isAvailable ? (
        <span style={{ fontSize: '1.25rem', fontWeight: 700, color: '#111827' }}>
          {value!.toFixed(1)}
        </span>
      ) : (
        <span style={{ fontSize: '0.875rem', color: '#9ca3af', fontStyle: 'italic' }}>
          Data not available
        </span>
      )}
    </div>
  )
}
