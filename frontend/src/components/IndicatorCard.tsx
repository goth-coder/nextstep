interface IndicatorCardProps {
  label: string
  value: number | null
  /** Key used to decide if a zero value is suspicious (e.g. 'iaa', 'ieg') */
  indicatorKey?: string
}

// Indicators where 0.0 is almost certainly missing data / imputation artifact
const ZERO_SUSPICIOUS_KEYS = new Set(['iaa', 'ieg', 'ips', 'ida', 'ipv', 'ipp', 'inde'])

function zeroTooltip(key: string): string {
  if (key === 'ieg') return 'IEG = 0 afeta 9% dos alunos — pode indicar ausência de avaliação de engajamento; verifique com o coordenador.'
  return 'Valor zero pode indicar dado ausente ou erro de registro — verifique o histórico do aluno.'
}

export default function IndicatorCard({ label, value, indicatorKey }: IndicatorCardProps) {
  const isAvailable = value !== null && value !== undefined
  const isZeroSuspicious =
    isAvailable && value === 0 && indicatorKey && ZERO_SUSPICIOUS_KEYS.has(indicatorKey)

  return (
    <div
      title={isZeroSuspicious ? zeroTooltip(indicatorKey!) : undefined}
      style={{
        padding: '0.875rem 1rem',
        background: '#fff',
        border: `1px solid ${isZeroSuspicious ? '#fbbf24' : '#e5e7eb'}`,
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
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <span style={{ fontSize: '1.25rem', fontWeight: 700, color: isZeroSuspicious ? '#92400e' : '#111827' }}>
            {value!.toFixed(1)}
          </span>
          {isZeroSuspicious && (
            <span
              title="Valor zero suspeito — pode ser dado ausente"
              style={{ fontSize: '0.85rem', cursor: 'help' }}
            >
              ⚠️
            </span>
          )}
        </div>
      ) : (
        <span style={{ fontSize: '0.875rem', color: '#9ca3af', fontStyle: 'italic' }}>
          Data not available
        </span>
      )}
    </div>
  )
}
