import type { StudentSummary } from '../types/student'
import RiskBadge from './RiskBadge'

interface StudentListItemProps {
  student: StudentSummary
  onClick: () => void
}

export default function StudentListItem({ student, onClick }: StudentListItemProps) {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      onClick()
    }
  }

  const riskPercent = student.risk_score != null ? `${(student.risk_score * 100).toFixed(1)}%` : '—'

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={handleKeyDown}
      aria-label={`View profile for ${student.display_name}`}
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr auto auto auto',
        alignItems: 'center',
        gap: '1rem',
        padding: '0.875rem 1.25rem',
        cursor: 'pointer',
        borderBottom: '1px solid #e5e7eb',
        transition: 'background 0.15s',
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.background = '#f9fafb'
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.background = 'transparent'
      }}
    >
      <span style={{ fontWeight: 500, color: '#111827' }}>{student.display_name}</span>
      <span style={{ color: '#6b7280', fontSize: '0.875rem' }}>{student.phase}</span>
      <RiskBadge risk_tier={student.risk_tier} />
      <span
        style={{
          fontVariantNumeric: 'tabular-nums',
          color: '#374151',
          fontSize: '0.875rem',
          minWidth: '3.5rem',
          textAlign: 'right',
        }}
      >
        {riskPercent}
      </span>
    </div>
  )
}
