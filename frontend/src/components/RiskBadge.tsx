import type { RiskTier } from '../types/student'
import { radius, riskColor, typography } from '../styles/theme'

interface RiskBadgeProps {
  risk_tier: RiskTier | null
}

const TIER_META: Record<RiskTier, { label: string; icon: string }> = {
  high:   { label: 'High Risk',   icon: '▲' },
  medium: { label: 'Medium Risk', icon: '●' },
  low:    { label: 'Low Risk',    icon: '▼' },
}

export default function RiskBadge({ risk_tier }: RiskBadgeProps) {
  if (!risk_tier) return <span style={{ color: '#9ca3af', fontSize: '0.75rem' }}>—</span>
  const { fg, bg } = riskColor(risk_tier)
  const { label, icon } = TIER_META[risk_tier]

  return (
    <span
      role="status"
      aria-label={label}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "0.3rem",
        paddingInline: "0.625rem",
        paddingBlock: "0.25rem",
        background: bg,
        color: fg,
        border: `1px solid ${fg}33`,
        borderRadius: radius.full,
        fontSize: typography.sizes.xs,
        fontWeight: 700,
        letterSpacing: "0.03em",
        whiteSpace: "nowrap",
        userSelect: "none",
      }}
    >
      <span aria-hidden style={{ fontSize: "0.6rem" }}>{icon}</span>
      {label}
    </span>
  )
}
