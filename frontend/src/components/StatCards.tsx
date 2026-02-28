import type { StudentSummary } from '../types/student'
import { colors, radius, shadows, typography } from '../styles/theme'

interface StatCardsProps {
  students: StudentSummary[]
}

interface StatCardData {
  label: string
  value: number
  accent: string
  bg: string
  icon: string
}

export default function StatCards({ students }: StatCardsProps) {
  const high = students.filter((s) => s.risk_tier === 'high').length
  const medium = students.filter((s) => s.risk_tier === 'medium').length
  const low = students.filter((s) => s.risk_tier === 'low').length

  const cards: StatCardData[] = [
    {
      label: 'Total Students',
      value: students.length,
      accent: colors.brandAccent,
      bg: '#EFF6FF',
      icon: '👥',
    },
    {
      label: 'High Risk',
      value: high,
      accent: colors.riskHigh,
      bg: colors.riskHighBg,
      icon: '🔴',
    },
    {
      label: 'Medium Risk',
      value: medium,
      accent: colors.riskMedium,
      bg: colors.riskMediumBg,
      icon: '🟡',
    },
    {
      label: 'Low Risk',
      value: low,
      accent: colors.riskLow,
      bg: colors.riskLowBg,
      icon: '🟢',
    },
  ]

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        gap: '1rem',
        marginBottom: '1.5rem',
      }}
    >
      {cards.map((card) => (
        <div
          key={card.label}
          style={{
            background: colors.white,
            border: `1px solid ${colors.gray200}`,
            borderRadius: radius.lg,
            padding: '1.25rem 1.5rem',
            boxShadow: shadows.sm,
            display: 'flex',
            flexDirection: 'column',
            gap: '0.5rem',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <span
              style={{
                fontSize: typography.sizes.xs,
                fontWeight: 600,
                color: colors.gray500,
                textTransform: 'uppercase',
                letterSpacing: '0.06em',
              }}
            >
              {card.label}
            </span>
            <span
              style={{
                background: card.bg,
                borderRadius: radius.full,
                width: '32px',
                height: '32px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1rem',
              }}
              aria-hidden
            >
              {card.icon}
            </span>
          </div>
          <span
            style={{
              fontSize: '2rem',
              fontWeight: 800,
              color: card.accent,
              lineHeight: 1,
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            {card.value.toLocaleString()}
          </span>
        </div>
      ))}
    </div>
  )
}
