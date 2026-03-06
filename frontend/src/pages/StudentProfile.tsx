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
}

const ZERO_SUSPICIOUS = new Set(['iaa', 'ieg', 'ips', 'ida', 'ipv', 'ipp', 'inde'])

function IndicatorCard({ label, indicatorKey, val }: {
  label: string
  indicatorKey: string
  val: number | null | undefined
}) {
  const [hovering, setHovering] = useState(false)
  const isAvailable = val !== null && val !== undefined
  const isZeroSuspect = isAvailable && (val as number) === 0 && ZERO_SUSPICIOUS.has(indicatorKey)
  const zeroTip = indicatorKey === 'ieg'
    ? 'IEG = 0 afeta ~9% dos alunos — pode indicar ausência de avaliação de engajamento. Verifique com o coordenador se houve avaliação no ciclo.'
    : 'Valor zero pode indicar dado ausente ou erro de registro. Verifique o histórico do aluno antes de tomar decisões baseadas neste indicador.'
  return (
    <div style={{
      position: 'relative',
      padding: '0.5rem 0.75rem',
      background: colors.white,
      border: `1px solid ${isZeroSuspect ? '#fbbf24' : colors.gray200}`,
      borderRadius: radius.md,
      boxShadow: shadows.sm,
      display: 'flex',
      flexDirection: 'column',
      gap: '0.1rem',
    }}>
      <span style={{ fontSize: '0.6rem', fontWeight: 700, color: colors.gray500, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {label.split(' — ')[0]}
      </span>
      <span style={{ fontSize: '0.6rem', color: colors.gray300 }}>
        {label.split(' — ')[1]}
      </span>
      {isAvailable ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', marginTop: '0.15rem' }}>
          <span style={{
            fontSize: typography.sizes.base, fontWeight: 800,
            color: isZeroSuspect ? '#92400e' : colors.brandPrimary,
            fontVariantNumeric: 'tabular-nums',
          }}>
            {indicatorKey === 'defasagem'
              ? `${(val as number) >= 0 ? '+' : ''}${val}`
              : (val as number).toFixed(1)}
          </span>
          {isZeroSuspect && (
            <span
              onMouseEnter={() => setHovering(true)}
              onMouseLeave={() => setHovering(false)}
              style={{ fontSize: '0.7rem', cursor: 'help' }}
            >⚠️</span>
          )}
        </div>
      ) : (
        <span style={{ fontSize: typography.sizes.xs, color: colors.gray300, fontStyle: 'italic' }}>N/A</span>
      )}
      {isZeroSuspect && hovering && (
        <div style={{
          position: 'absolute', bottom: 'calc(100% + 6px)', left: 0,
          zIndex: 300, background: '#1e293b', color: 'white',
          borderRadius: '0.5rem', padding: '0.5rem 0.75rem',
          fontSize: '0.68rem', lineHeight: 1.5,
          width: '220px', boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
          pointerEvents: 'none',
        }}>
          <span style={{ color: '#fbbf24', fontWeight: 700 }}>⚠️ Dado suspeito</span><br />
          {zeroTip}
        </div>
      )}
    </div>
  )
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
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
            height: 'calc(100vh - 6rem)',
            minHeight: 0,
          }}
        >
          {/* ── Compact header ── */}
          <div
            style={{
              background: colors.white,
              border: `1px solid ${colors.gray200}`,
              borderRadius: radius.lg,
              padding: '0.625rem 1.25rem',
              boxShadow: shadows.sm,
              display: 'flex',
              alignItems: 'center',
              gap: '1rem',
              flexWrap: 'wrap',
            }}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              <h1
                style={{
                  fontSize: typography.sizes.lg,
                  fontWeight: 800,
                  color: colors.brandPrimary,
                  margin: 0,
                  letterSpacing: '-0.02em',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                {student.display_name}
              </h1>
              <p style={{ color: colors.gray500, fontSize: typography.sizes.xs, margin: '0.1rem 0 0' }}>
                {student.phase}{student.class_group && ` · Turma ${student.class_group}`}{student.fase_num !== null && ` · Fase ${student.fase_num}`}
              </p>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexShrink: 0 }}>
              {/* Defasagem — highlighted next to the name */}
              {student.indicators.defasagem !== null && student.indicators.defasagem !== undefined && (() => {
                const def = Math.round(student.indicators.defasagem as number)
                const isLag = def > 0
                return (
                  <div style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center',
                    background: isLag ? '#fef3c7' : '#dcfce7',
                    border: `1px solid ${isLag ? '#fbbf24' : '#86efac'}`,
                    borderRadius: radius.md,
                    padding: '0.25rem 0.75rem',
                    gap: '0.05rem',
                  }}>
                    <span style={{ fontSize: '0.6rem', fontWeight: 700, color: isLag ? '#92400e' : '#166534', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Defasagem</span>
                    <span style={{ fontSize: typography.sizes.base, fontWeight: 800, color: isLag ? '#92400e' : '#166534', fontVariantNumeric: 'tabular-nums' }}>
                      {def > 0 ? '+' : ''}{def} fase{Math.abs(def) !== 1 ? 's' : ''}
                    </span>
                  </div>
                )
              })()}
              <span style={{
                fontVariantNumeric: 'tabular-nums', fontWeight: 800,
                fontSize: typography.sizes.xl, color: colors.brandPrimary,
                letterSpacing: '-0.02em',
              }}>
                {student.risk_score != null ? `${(student.risk_score * 100).toFixed(1)}%` : '—'}
              </span>
              <span style={{ fontSize: typography.sizes.xs, color: colors.gray500 }}>risco</span>
              <RiskBadge risk_tier={student.risk_tier} />
            </div>
          </div>

          {/* ── Body: left col = indicators + advice, right col = radar ── */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 320px',
              gap: '0.75rem',
              flex: 1,
              minHeight: 0,
              overflow: 'hidden',
            }}
          >
            {/* Left: indicators + advice stacked */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', minHeight: 0, overflow: 'hidden' }}>
              <section
                aria-label="Student indicators"
                style={{
                  background: colors.white,
                  border: `1px solid ${colors.gray200}`,
                  borderRadius: radius.lg,
                  padding: '0.875rem 1rem',
                  boxShadow: shadows.sm,
                  flexShrink: 0,
                }}
              >
              <h2 style={{
                fontSize: typography.sizes.xs, fontWeight: 700, color: colors.gray500,
                textTransform: 'uppercase', letterSpacing: '0.07em',
                marginBottom: '0.625rem', marginTop: 0,
              }}>
                Performance Indicators
              </h2>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
                  gap: '0.5rem',
                }}
              >
                {(Object.keys(INDICATOR_LABELS) as Array<keyof typeof INDICATOR_LABELS>).map(
                  (key) => (
                    <IndicatorCard
                      key={key}
                      indicatorKey={key}
                      label={INDICATOR_LABELS[key]}
                      val={student.indicators[key as keyof typeof student.indicators] as number | null | undefined}
                    />
                  ),
                )}
              </div>
              </section>

              {/* AI Advice below indicators */}
              <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
                <AdvicePanel studentId={student.student_id} riskTier={student.risk_tier} />
              </div>
            </div>

            {/* Right: radar only */}
            <aside
              style={{
                background: colors.white,
                border: `1px solid ${colors.gray200}`,
                borderRadius: radius.lg,
                padding: '0.875rem 1rem',
                boxShadow: shadows.sm,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '0.5rem',
                overflow: 'hidden',
              }}
            >
              <h2 style={{
                fontSize: typography.sizes.xs, fontWeight: 700, color: colors.gray500,
                textTransform: 'uppercase', letterSpacing: '0.07em',
                margin: 0, alignSelf: 'flex-start',
              }}>
                Indicator Radar
              </h2>
              <IndicatorRadar
                indicators={student.indicators as unknown as Record<string, number | null>}
                labels={INDICATOR_LABELS}
              />
            </aside>
          </div>
        </div>
      )}
    </AppShell>
  )
}
