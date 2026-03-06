import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import AppShell from '../components/AppShell'
import ErrorState from '../components/ErrorState'
import RiskBadge from '../components/RiskBadge'
import StatCards from '../components/StatCards'
import { getOrFetchStudents, invalidateStudentCache } from '../services/studentCache'
import { colors, radius, shadows, typography } from '../styles/theme'
import type { RiskTier, StudentSummary } from '../types/student'

type Status = 'loading' | 'success' | 'error'

const PAGE_SIZE_OPTIONS = [25, 50, 100]

const RISK_TIERS: { value: RiskTier | 'all'; label: string; color: string; bg: string }[] = [
  { value: 'all',    label: 'All',    color: colors.gray600, bg: colors.gray100 },
  { value: 'high',   label: 'High',   color: colors.riskHigh, bg: colors.riskHighBg },
  { value: 'medium', label: 'Medium', color: colors.riskMedium, bg: colors.riskMediumBg },
  { value: 'low',    label: 'Low',    color: colors.riskLow, bg: colors.riskLowBg },
]

export default function Dashboard() {
  const navigate = useNavigate()
  const [status, setStatus] = useState<Status>('loading')
  const [students, setStudents] = useState<StudentSummary[]>([])
  const [errorMsg, setErrorMsg] = useState('')

  // Filters
  const [searchQuery, setSearchQuery] = useState('')
  const [riskFilter, setRiskFilter] = useState<RiskTier | 'all'>('all')

  // Pagination
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(25)

  const fetchStudents = useCallback(async () => {
    setStatus('loading')
    try {
      const data = await getOrFetchStudents()
      setStudents(data.students)
      setStatus('success')
    } catch {
      setErrorMsg('Failed to load students. Please check the API connection.')
      setStatus('error')
    }
  }, [])

  const handleRetry = useCallback(() => {
    invalidateStudentCache()
    fetchStudents()
  }, [fetchStudents])

  useEffect(() => { fetchStudents() }, [fetchStudents])

  // Apply filters
  const filtered = useMemo(() => {
    const q = searchQuery.toLowerCase()
    return students.filter((s) => {
      if (q && !s.display_name.toLowerCase().includes(q) && !s.class_group?.toLowerCase().includes(q))
        return false
      if (riskFilter !== 'all' && s.risk_tier !== riskFilter) return false
      return true
    })
  }, [students, searchQuery, riskFilter])

  // Reset to page 1 on filter change
  useEffect(() => { setPage(1) }, [searchQuery, riskFilter])

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize))
  const currentPage = Math.min(page, totalPages)
  const paginated = filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize)

  const hasFilters = searchQuery !== '' || riskFilter !== 'all'

  const clearFilters = () => {
    setSearchQuery('')
    setRiskFilter('all')
  }

  return (
    <AppShell>
      {/* Page heading */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h1 style={{
          fontSize: typography.sizes['2xl'], fontWeight: 800,
          color: colors.brandPrimary, margin: 0, letterSpacing: '-0.02em',
        }}>
          Student Risk Dashboard
        </h1>
        <p style={{ color: colors.gray500, marginTop: '0.25rem', fontSize: typography.sizes.sm }}>
          Sorted by school delay risk, highest first
        </p>
      </div>

      {status === 'loading' && (
        <div role="status" aria-live="polite"
          style={{ textAlign: 'center', padding: '5rem', color: colors.gray500 }}>
          <div style={{ fontSize: '2rem', marginBottom: '0.75rem' }}>⏳</div>
          Loading students…
        </div>
      )}

      {status === 'error' && <ErrorState message={errorMsg} onRetry={handleRetry} />}

      {status === 'success' && (
        <>
          <StatCards students={students} />

          {/* ── Filter bar ─────────────────────────────────────────────── */}
          <div style={{
            background: colors.white,
            border: `1px solid ${colors.gray200}`,
            borderRadius: radius.lg,
            padding: '1rem 1.25rem',
            marginBottom: '1rem',
            display: 'flex',
            flexWrap: 'wrap',
            gap: '0.75rem',
            alignItems: 'center',
            boxShadow: shadows.sm,
          }}>
            {/* Search */}
            <div style={{ position: 'relative', flex: '1 1 220px', minWidth: '180px' }}>
              <input
                type="search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search name or class…"
                aria-label="Search students by name or class"
                style={{
                  width: '100%', paddingLeft: '0.75rem', paddingRight: '0.75rem',
                  paddingTop: '0.5rem', paddingBottom: '0.5rem',
                  fontSize: typography.sizes.sm,
                  border: `1.5px solid ${colors.gray200}`, borderRadius: radius.md,
                  outline: 'none', background: colors.white, color: colors.gray900,
                  boxSizing: 'border-box',
                }}
                onFocus={(e) => (e.currentTarget.style.borderColor = colors.brandAccent)}
                onBlur={(e) => (e.currentTarget.style.borderColor = colors.gray200)}
              />
            </div>

            {/* Risk tier pills */}
            <div style={{ display: 'flex', gap: '0.375rem', flexWrap: 'wrap' }} role="group" aria-label="Filter by risk level">
              {RISK_TIERS.map((tier) => {
                const active = riskFilter === tier.value
                return (
                  <button
                    key={tier.value}
                    onClick={() => setRiskFilter(tier.value)}
                    aria-pressed={active}
                    style={{
                      padding: '0.375rem 0.75rem',
                      fontSize: typography.sizes.xs,
                      fontWeight: active ? 700 : 500,
                      border: `1.5px solid ${active ? tier.color : colors.gray200}`,
                      borderRadius: radius.full,
                      background: active ? tier.bg : colors.white,
                      color: active ? tier.color : colors.gray500,
                      cursor: 'pointer',
                      transition: 'all 0.12s',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {tier.label}
                  </button>
                )
              })}
            </div>

            {/* Clear */}
            {hasFilters && (
              <button
                onClick={clearFilters}
                aria-label="Clear all filters"
                style={{
                  padding: '0.375rem 0.75rem', fontSize: typography.sizes.xs,
                  color: colors.gray600, background: colors.gray100,
                  border: `1px solid ${colors.gray200}`,
                  borderRadius: radius.md, cursor: 'pointer',
                  marginLeft: 'auto',
                }}
              >
                ✕ Clear
              </button>
            )}
          </div>

          {/* Results summary + page-size */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem',
          }}>
            <span style={{ fontSize: typography.sizes.sm, color: colors.gray500 }}>
              Showing{' '}
              <strong style={{ color: colors.gray700 }}>
                {filtered.length === 0 ? 0 : (currentPage - 1) * pageSize + 1}–{Math.min(currentPage * pageSize, filtered.length)}
              </strong>{' '}
              of <strong style={{ color: colors.gray700 }}>{filtered.length}</strong>
              {hasFilters ? ' filtered' : ''} students
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <label htmlFor="page-size" style={{ fontSize: typography.sizes.xs, color: colors.gray500 }}>
                Per page:
              </label>
              <select
                id="page-size"
                value={pageSize}
                onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1) }}
                style={{
                  fontSize: typography.sizes.xs,
                  border: `1px solid ${colors.gray200}`,
                  borderRadius: radius.sm, padding: '0.25rem 0.5rem',
                  background: colors.white, color: colors.gray700, cursor: 'pointer',
                }}
              >
                {PAGE_SIZE_OPTIONS.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
          </div>

          {/* Student table */}
          {filtered.length === 0 ? (
            <div style={{
              textAlign: 'center', padding: '4rem', color: colors.gray500,
              background: colors.white, border: `1px solid ${colors.gray200}`,
              borderRadius: radius.lg,
            }}>
              No students match your filters.
            </div>
          ) : (
            <section aria-label="Student list" style={{
              background: colors.white, border: `1px solid ${colors.gray200}`,
              borderRadius: radius.lg, overflow: 'hidden', boxShadow: shadows.sm,
            }}>
              {/* Header */}
              <div aria-hidden style={{
                display: 'grid',
                gridTemplateColumns: '1fr 110px 130px 80px',
                gap: '1rem', padding: '0.625rem 1.25rem',
                background: colors.gray50, borderBottom: `1px solid ${colors.gray200}`,
                fontSize: typography.sizes.xs, fontWeight: 700,
                color: colors.gray500, textTransform: 'uppercase', letterSpacing: '0.06em',
              }}>
                <span>Name</span>
                <span>Class</span>
                <span>Risk Level</span>
                <span style={{ textAlign: 'right' }}>Score</span>
              </div>

              {paginated.map((student, idx) => (
                <StudentRow
                  key={student.student_id}
                  student={student}
                  isLast={idx === paginated.length - 1}
                  onClick={() => navigate(`/students/${student.student_id}`)}
                />
              ))}
            </section>
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              gap: '0.5rem', marginTop: '1.25rem',
            }}>
              <PageButton label="← Prev" onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={currentPage === 1} />
              {pageNumbers(currentPage, totalPages).map((n, i) =>
                n === '…' ? (
                  <span key={`e${i}`} style={{ color: colors.gray500, padding: '0 0.25rem' }}>…</span>
                ) : (
                  <PageButton key={n} label={String(n)} onClick={() => setPage(Number(n))} active={n === currentPage} />
                ),
              )}
              <PageButton label="Next →" onClick={() => setPage((p) => Math.min(totalPages, p + 1))} disabled={currentPage === totalPages} />
            </div>
          )}
        </>
      )}
    </AppShell>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StudentRow({ student, isLast, onClick }: {
  student: StudentSummary; isLast: boolean; onClick: () => void
}) {
  const [hovered, setHovered] = useState(false)
  return (
    <div
      role="button" tabIndex={0} onClick={onClick}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick() } }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      aria-label={`View profile for ${student.display_name}`}
      style={{
        display: 'grid', gridTemplateColumns: '1fr 110px 130px 80px',
        alignItems: 'center', gap: '1rem', padding: '0.875rem 1.25rem',
        cursor: 'pointer',
        borderBottom: isLast ? 'none' : `1px solid ${colors.gray100}`,
        background: hovered ? colors.gray50 : colors.white,
        transition: 'background 0.12s',
      }}
    >
      <span style={{ fontWeight: 600, color: colors.gray900, fontSize: typography.sizes.sm }}>
        {student.display_name}
      </span>
      <span style={{
        color: colors.gray600, fontSize: typography.sizes.sm,
        background: colors.gray100, borderRadius: radius.sm,
        padding: '0.15rem 0.5rem', display: 'inline-block', fontWeight: 500,
      }}>
        {student.class_group || '—'}
      </span>
      <RiskBadge risk_tier={student.risk_tier} />
      <span style={{
        fontVariantNumeric: 'tabular-nums', fontWeight: 700,
        color: colors.gray700, fontSize: typography.sizes.sm,
        textAlign: 'right',
      }}>
        {student.risk_score != null ? `${(student.risk_score * 100).toFixed(1)}%` : '—'}
      </span>
    </div>
  )
}

function PageButton({ label, onClick, disabled = false, active = false }: {
  label: string; onClick: () => void; disabled?: boolean; active?: boolean
}) {
  return (
    <button
      onClick={onClick} disabled={disabled}
      aria-current={active ? 'page' : undefined}
      style={{
        padding: '0.375rem 0.75rem', fontSize: typography.sizes.sm,
        fontWeight: active ? 700 : 400,
        border: `1.5px solid ${active ? colors.brandAccent : colors.gray200}`,
        borderRadius: radius.md,
        background: active ? colors.brandAccent : colors.white,
        color: active ? colors.white : disabled ? colors.gray300 : colors.gray700,
        cursor: disabled ? 'not-allowed' : 'pointer',
        minWidth: '2.5rem',
      }}
    >
      {label}
    </button>
  )
}

function pageNumbers(current: number, total: number): (number | '…')[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1)
  const pages: (number | '…')[] = [1]
  if (current > 3) pages.push('…')
  for (let p = Math.max(2, current - 1); p <= Math.min(total - 1, current + 1); p++) pages.push(p)
  if (current < total - 2) pages.push('…')
  pages.push(total)
  return pages
}
