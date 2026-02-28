import { useCallback, useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import AdvicePanel from '../components/AdvicePanel'
import AppShell from '../components/AppShell'
import ErrorState from '../components/ErrorState'
import IndicatorRadar from '../components/IndicatorRadar'
import RiskBadge from '../components/RiskBadge'
import { getStudent } from '../services/api'
import { colors, radius, shadows, typography } from '../styles/theme'
import type { StudentDetail } from '../types/student'

type Status = 'loading' | 'success' | 'error'

const INDICATOR_LABELS: Record<string, string> = {
  iaa: 'IAA — Academic Performance',
  ieg: 'IEG — Engagement',
  ips: 'IPS — Psychosocial',
  ida: 'IDA — Self-sufficiency',
  ipv: 'IPV — Life Vision',
  ipp: 'IPP — Psychopedagogical',
  inde: 'INDE — Development Index',
  defasagem: 'Defasagem — School Lag',
}

export default function StudentProfile() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [status, setStatus] = useState<Status>('loading')
  const [student, setStudent] = useState<StudentDetail | null>(null)
  const [errorMsg, setErrorMsg] = useState('')

  const fetchStudent = useCallback(async () => {
    if (!id) return
    setStatus('loading')
    try {
      const data = await getStudent(Number(id))
      setStudent(data)
      setStatus('success')
    } catch (err: unknown) {
      const httpStatus = (err as { response?: { status?: number } })?.response?.status
      setErrorMsg(
        httpStatus === 404
          ? `Student #${id} not found.`
          : 'Failed to load student profile. Please try again.',
      )
      setStatus('error')
    }
  }, [id])

  useEffect(() => {
    fetchStudent()
  }, [fetchStudent])

  return (
    <AppShell>
      {/* Back link */}
      <button
        onClick={() => navigate('/')}
        style={{
          background: 'none',
          border: 'none',
          color: colors.brandAccent,
          cursor: 'pointer',
          padding: '0.25rem 0',
          fontSize: typography.sizes.sm,
          fontWeight: 600,
          marginBottom: '1.25rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.375rem',
        }}
      >
        ← Back to Dashboard
      </button>

      {status === 'loading' && (
        <div
          role="status"
          aria-live="polite"
          style={{ textAlign: 'center', padding: '5rem', color: colors.gray500 }}
        >
          <div style={{ fontSize: '2rem', marginBottom: '0.75rem' }}>⏳</div>
          Loading profile…
        </div>
      )}

      {status === 'error' && <ErrorState message={errorMsg} onRetry={fetchStudent} />}

      {status === 'success' && student && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr',
            gap: '1.25rem',
            maxWidth: '960px',
          }}
        >
          {/* Profile header card */}
          <div
            style={{
              background: colors.white,
              border: `1px solid ${colors.gray200}`,
              borderRadius: radius.xl,
              padding: '1.75rem 2rem',
              boxShadow: shadows.md,
              display: 'flex',
              alignItems: 'flex-start',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '1rem',
            }}
          >
            {/* Left: identity */}
            <div>
              <h1
                style={{
                  fontSize: typography.sizes['2xl'],
                  fontWeight: 800,
                  color: colors.brandPrimary,
                  margin: 0,
                  letterSpacing: '-0.02em',
                }}
              >
                {student.display_name}
              </h1>
              <p
                style={{
                  color: colors.gray500,
                  fontSize: typography.sizes.sm,
                  marginTop: '0.375rem',
                  marginBottom: 0,
                }}
              >
                {student.phase}
                {student.class_group && ` · Turma ${student.class_group}`}
                {student.fase_num !== null && ` · Fase ${student.fase_num}`}
              </p>
            </div>

            {/* Right: risk */}
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-end',
                gap: '0.5rem',
              }}
            >
              <RiskBadge risk_tier={student.risk_tier} />
              <span
                style={{
                  fontVariantNumeric: 'tabular-nums',
                  fontWeight: 800,
                  fontSize: typography.sizes['3xl'],
                  color: colors.brandPrimary,
                  letterSpacing: '-0.02em',
                }}
              >
                {(student.risk_score * 100).toFixed(1)}%
              </span>
              <span style={{ fontSize: typography.sizes.xs, color: colors.gray500 }}>
                dropout risk score
              </span>
            </div>
          </div>

          {/* Two-column: indicators + radar */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: '1.25rem',
            }}
          >
            {/* Indicator cards */}
            <section aria-label="Student indicators">
              <h2
                style={{
                  fontSize: typography.sizes.xs,
                  fontWeight: 700,
                  color: colors.gray500,
                  textTransform: 'uppercase',
                  letterSpacing: '0.07em',
                  marginBottom: '0.75rem',
                  marginTop: 0,
                }}
              >
                Performance Indicators
              </h2>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
                  gap: '0.625rem',
                }}
              >
                {(Object.keys(INDICATOR_LABELS) as Array<keyof typeof INDICATOR_LABELS>).map(
                  (key) => {
                    const val = student.indicators[key as keyof typeof student.indicators]
                    const isAvailable = val !== null && val !== undefined
                    return (
                      <div
                        key={key}
                        style={{
                          padding: '0.875rem 1rem',
                          background: colors.white,
                          border: `1px solid ${colors.gray200}`,
                          borderRadius: radius.md,
                          boxShadow: shadows.sm,
                          display: 'flex',
                          flexDirection: 'column',
                          gap: '0.25rem',
                        }}
                      >
                        <span
                          style={{
                            fontSize: typography.sizes.xs,
                            fontWeight: 700,
                            color: colors.gray500,
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em',
                          }}
                        >
                          {INDICATOR_LABELS[key].split(' — ')[0]}
                        </span>
                        <span
                          style={{
                            fontSize: typography.sizes.xs,
                            color: colors.gray500,
                          }}
                        >
                          {INDICATOR_LABELS[key].split(' — ')[1]}
                        </span>
                        {isAvailable ? (
                          <span
                            style={{
                              fontSize: typography.sizes.xl,
                              fontWeight: 800,
                              color: colors.brandPrimary,
                              fontVariantNumeric: 'tabular-nums',
                            }}
                          >
                            {key === 'defasagem'
                              ? `${(val as number) >= 0 ? '+' : ''}${val}`
                              : (val as number).toFixed(1)}
                          </span>
                        ) : (
                          <span
                            style={{
                              fontSize: typography.sizes.sm,
                              color: colors.gray300,
                              fontStyle: 'italic',
                            }}
                          >
                            N/A
                          </span>
                        )}
                      </div>
                    )
                  },
                )}
              </div>
            </section>

            {/* Radar chart */}
            <aside
              style={{
                background: colors.white,
                border: `1px solid ${colors.gray200}`,
                borderRadius: radius.lg,
                padding: '1.5rem',
                boxShadow: shadows.sm,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '0.75rem',
              }}
            >
              <h2
                style={{
                  fontSize: typography.sizes.xs,
                  fontWeight: 700,
                  color: colors.gray500,
                  textTransform: 'uppercase',
                  letterSpacing: '0.07em',
                  margin: 0,
                  alignSelf: 'flex-start',
                }}
              >
                Indicator Radar
              </h2>
              <IndicatorRadar
                indicators={student.indicators as unknown as Record<string, number | null>}
                labels={INDICATOR_LABELS}
              />
            </aside>
          </div>

          {/* AI Advice panel */}
          <AdvicePanel studentId={student.student_id} riskTier={student.risk_tier} />
        </div>
      )}
    </AppShell>
  )
}
