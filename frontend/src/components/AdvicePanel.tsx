import { useEffect, useState } from 'react'
import { getAdvice } from '../services/api'
import type { PedagogicalAdvice, RiskTier } from '../types/student'

interface AdvicePanelProps {
  studentId: number
  riskTier: RiskTier
}

export default function AdvicePanel({ studentId, riskTier }: AdvicePanelProps) {
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState<PedagogicalAdvice | null>(null)

  useEffect(() => {
    // Do not show advice for low-risk students
    if (riskTier === 'low') return

    let cancelled = false
    setLoading(true)
    setData(null)

    getAdvice(studentId)
      .then((result) => {
        if (!cancelled) {
          setData(result)
        }
      })
      .catch(() => {
        // API always returns 200 with is_fallback=true on error — unexpected here
        // Silently ignore to avoid duplicate error UI
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [studentId, riskTier])

  // Low-risk students: no panel
  if (riskTier === 'low') return null

  return (
    <section
      aria-label="Pedagogical suggestion"
      style={{
        background: '#fff',
        border: '1px solid #e5e7eb',
        borderRadius: '0.5rem',
        padding: '1.25rem 1.5rem',
      }}
    >
      <h2
        style={{
          fontSize: '0.875rem',
          fontWeight: 600,
          color: '#6b7280',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          marginTop: 0,
          marginBottom: '0.75rem',
        }}
      >
        Pedagogical Suggestion
      </h2>

      {loading && (
        <p role="status" aria-live="polite" style={{ color: '#9ca3af', fontStyle: 'italic', margin: 0 }}>
          Generating suggestion…
        </p>
      )}

      {!loading && data && (
        <p
          style={{
            color: data.is_fallback ? '#9ca3af' : '#374151',
            fontStyle: data.is_fallback ? 'italic' : 'normal',
            lineHeight: 1.65,
            margin: 0,
            whiteSpace: 'pre-wrap',
          }}
        >
          {data.advice}
        </p>
      )}
    </section>
  )
}
