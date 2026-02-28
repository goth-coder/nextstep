import { useCallback, useEffect, useRef, useState } from 'react'
import { colors, radius, shadows, typography } from '../styles/theme'

interface SearchBarProps {
  onSearch: (query: string, classGroup: string) => void
  classGroups: string[]
}

function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])
  return debounced
}

export default function SearchBar({ onSearch, classGroups }: SearchBarProps) {
  const [query, setQuery] = useState('')
  const [classGroup, setClassGroup] = useState('')
  const debouncedQuery = useDebounce(query, 300)

  // Stable callback ref — prevent stale closure in useEffect
  const onSearchRef = useRef(onSearch)
  useEffect(() => { onSearchRef.current = onSearch }, [onSearch])

  useEffect(() => {
    onSearchRef.current(debouncedQuery, classGroup)
  }, [debouncedQuery, classGroup])

  const handleClear = useCallback(() => {
    setQuery('')
    setClassGroup('')
  }, [])

  const hasFilter = query !== '' || classGroup !== ''

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        flexWrap: 'wrap',
        marginBottom: '1rem',
      }}
    >
      {/* Search input */}
      <div style={{ position: 'relative', flex: '1 1 240px', minWidth: '200px' }}>
        <span
          aria-hidden
          style={{
            position: 'absolute',
            left: '0.75rem',
            top: '50%',
            transform: 'translateY(-50%)',
            color: colors.gray500,
            fontSize: '0.875rem',
            pointerEvents: 'none',
          }}
        >
          🔍
        </span>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by name…"
          aria-label="Search students by name"
          style={{
            width: '100%',
            paddingLeft: '2.25rem',
            paddingRight: '0.75rem',
            paddingTop: '0.5rem',
            paddingBottom: '0.5rem',
            fontSize: typography.sizes.sm,
            border: `1.5px solid ${colors.gray200}`,
            borderRadius: radius.md,
            outline: 'none',
            background: colors.white,
            color: colors.gray900,
            boxSizing: 'border-box',
            boxShadow: shadows.sm,
            transition: 'border-color 0.15s',
          }}
          onFocus={(e) =>
            (e.currentTarget.style.borderColor = colors.brandAccent)
          }
          onBlur={(e) => (e.currentTarget.style.borderColor = colors.gray200)}
        />
      </div>

      {/* Class group filter */}
      {classGroups.length > 0 && (
        <select
          value={classGroup}
          onChange={(e) => setClassGroup(e.target.value)}
          aria-label="Filter by class group"
          style={{
            padding: '0.5rem 2rem 0.5rem 0.75rem',
            fontSize: typography.sizes.sm,
            border: `1.5px solid ${colors.gray200}`,
            borderRadius: radius.md,
            outline: 'none',
            background: colors.white,
            color: classGroup ? colors.gray900 : colors.gray500,
            boxShadow: shadows.sm,
            cursor: 'pointer',
            flex: '0 0 auto',
          }}
          onFocus={(e) =>
            (e.currentTarget.style.borderColor = colors.brandAccent)
          }
          onBlur={(e) => (e.currentTarget.style.borderColor = colors.gray200)}
        >
          <option value="">All classes</option>
          {classGroups.map((g) => (
            <option key={g} value={g}>
              Turma {g}
            </option>
          ))}
        </select>
      )}

      {/* Clear filters */}
      {hasFilter && (
        <button
          onClick={handleClear}
          aria-label="Clear filters"
          style={{
            padding: '0.5rem 0.875rem',
            fontSize: typography.sizes.sm,
            color: colors.gray600,
            background: colors.gray100,
            border: `1px solid ${colors.gray200}`,
            borderRadius: radius.md,
            cursor: 'pointer',
            transition: 'background 0.15s',
          }}
          onMouseEnter={(e) =>
            ((e.currentTarget as HTMLButtonElement).style.background =
              colors.gray200)
          }
          onMouseLeave={(e) =>
            ((e.currentTarget as HTMLButtonElement).style.background =
              colors.gray100)
          }
        >
          ✕ Clear
        </button>
      )}
    </div>
  )
}
