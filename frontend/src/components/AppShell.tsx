import type { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { colors, shadows, typography } from '../styles/theme'

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
        flexDirection: 'column',
      }}
    >
      {/* Topbar */}
      <header
        style={{
          background: colors.brandPrimary,
          boxShadow: shadows.md,
          position: 'sticky',
          top: 0,
          zIndex: 100,
        }}
      >
        <div
          style={{
            maxWidth: '1280px',
            margin: '0 auto',
            padding: '0 1.5rem',
            height: '60px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
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
            }}
          >
            <div
              aria-hidden
              style={{
                width: '32px',
                height: '32px',
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

          {/* Nav links */}
          <nav style={{ display: 'flex', gap: '0.25rem' }}>
            <NavLink to="/" active={location.pathname === '/'}>
              Dashboard
            </NavLink>
            <NavLink to="/model" active={location.pathname === '/model'}>
              Model
            </NavLink>
          </nav>
        </div>
      </header>

      {/* Page content */}
      <main
        style={{
          flex: 1,
          maxWidth: '1280px',
          width: '100%',
          margin: '0 auto',
          padding: '2rem 1.5rem',
          boxSizing: 'border-box',
        }}
      >
        {children}
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: `1px solid ${colors.gray200}`,
          background: colors.white,
          padding: '0.875rem 1.5rem',
          textAlign: 'center',
          fontSize: typography.sizes.xs,
          color: colors.gray500,
        }}
      >
        NextStep · FIAP — Análise de Risco de Defasagem Escolar · Dados PEDE 2022–2024
      </footer>
    </div>
  )
}

function NavLink({
  to,
  active,
  children,
}: {
  to: string
  active: boolean
  children: ReactNode
}) {
  return (
    <Link
      to={to}
      style={{
        color: active ? colors.white : 'rgba(255,255,255,0.65)',
        fontWeight: active ? 600 : 400,
        fontSize: typography.sizes.sm,
        textDecoration: 'none',
        padding: '0.375rem 0.75rem',
        borderRadius: '0.375rem',
        background: active ? 'rgba(255,255,255,0.12)' : 'transparent',
        transition: 'background 0.15s, color 0.15s',
      }}
    >
      {children}
    </Link>
  )
}
