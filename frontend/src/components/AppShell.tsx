import type { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { colors, shadows, typography } from '../styles/theme'

const SIDEBAR_W = 200

interface AppShellProps {
  children: ReactNode
}

export default function AppShell({ children }: AppShellProps) {
  const location = useLocation()

  return (
    <div
      style={{
        minHeight: '100vh',
        background: colors.surface,
        fontFamily: typography.fontFamily,
        display: 'flex',
        flexDirection: 'row',
      }}
    >
      {/* ── Left sidebar ── */}
      <aside
        style={{
          width: `${SIDEBAR_W}px`,
          minWidth: `${SIDEBAR_W}px`,
          background: colors.brandPrimary,
          boxShadow: shadows.md,
          position: 'sticky',
          top: 0,
          height: '100vh',
          display: 'flex',
          flexDirection: 'column',
          zIndex: 100,
          overflowY: 'auto',
        }}
      >
        {/* Logo */}
        <Link
          to="/"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.625rem',
            textDecoration: 'none',
            padding: '1.375rem 1.25rem 1.125rem',
            borderBottom: '1px solid rgba(255,255,255,0.08)',
          }}
        >
          <div
            aria-hidden
            style={{
              width: '32px',
              height: '32px',
              flexShrink: 0,
              background: colors.brandAccent,
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '1rem',
              fontWeight: 800,
              color: colors.white,
              letterSpacing: '-0.02em',
            }}
          >
            N
          </div>
          <span
            style={{
              color: colors.white,
              fontWeight: 700,
              fontSize: typography.sizes.lg,
              letterSpacing: '-0.01em',
            }}
          >
            NextStep
          </span>
        </Link>

        {/* Nav */}
        <nav
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '0.25rem',
            padding: '1rem 0.75rem',
            flex: 1,
          }}
        >
          <NavLink to="/" active={location.pathname === '/'} icon="⊞">
            Dashboard
          </NavLink>
          <NavLink to="/model" active={location.pathname === '/model'} icon="◈">
            Model
          </NavLink>
        </nav>

        {/* Footer caption */}
        <div
          style={{
            padding: '1rem 1.25rem',
            fontSize: '0.65rem',
            color: 'rgba(255,255,255,0.35)',
            lineHeight: 1.5,
            borderTop: '1px solid rgba(255,255,255,0.08)',
          }}
        >
          FIAP · PEDE 2022–2024<br />
          Defasagem Escolar
        </div>
      </aside>

      {/* ── Page content ── */}
      <main
        style={{
          flex: 1,
          minWidth: 0,
          padding: '1.25rem 1.5rem',
          boxSizing: 'border-box',
          overflowX: 'hidden',
        }}
      >
        {children}
      </main>
    </div>
  )
}

function NavLink({
  to,
  active,
  icon,
  children,
}: {
  to: string
  active: boolean
  icon: string
  children: ReactNode
}) {
  return (
    <Link
      to={to}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        color: active ? colors.white : 'rgba(255,255,255,0.6)',
        fontWeight: active ? 600 : 400,
        fontSize: typography.sizes.sm,
        textDecoration: 'none',
        padding: '0.5rem 0.75rem',
        borderRadius: '0.4rem',
        background: active ? 'rgba(255,255,255,0.13)' : 'transparent',
        transition: 'background 0.15s, color 0.15s',
      }}
    >
      <span style={{ fontSize: '0.85rem', opacity: 0.8 }}>{icon}</span>
      {children}
    </Link>
  )
}
