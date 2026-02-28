// Design tokens — NextStep color palette, typography, shadows, radius

export const colors = {
  // Brand
  brandPrimary: '#1E3A5F',
  brandAccent: '#3B82F6',
  brandAccentHover: '#2563EB',

  // Risk tiers
  riskHigh: '#DC2626',
  riskHighBg: '#FEF2F2',
  riskMedium: '#D97706',
  riskMediumBg: '#FFFBEB',
  riskLow: '#16A34A',
  riskLowBg: '#F0FDF4',

  // Surface / neutral
  surface: '#F8FAFC',
  white: '#FFFFFF',
  gray50: '#F9FAFB',
  gray100: '#F3F4F6',
  gray200: '#E5E7EB',
  gray300: '#D1D5DB',
  gray500: '#6B7280',
  gray600: '#4B5563',
  gray700: '#374151',
  gray900: '#111827',
} as const

export const shadows = {
  sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
  md: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
  lg: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
} as const

export const radius = {
  sm: '0.375rem',
  md: '0.5rem',
  lg: '0.75rem',
  xl: '1rem',
  full: '9999px',
} as const

export const typography = {
  fontFamily:
    "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  sizes: {
    xs: '0.75rem',
    sm: '0.875rem',
    base: '1rem',
    lg: '1.125rem',
    xl: '1.25rem',
    '2xl': '1.5rem',
    '3xl': '1.875rem',
  },
} as const

export type RiskTierColor = 'high' | 'medium' | 'low'

export function riskColor(tier: RiskTierColor) {
  return {
    high: { fg: colors.riskHigh, bg: colors.riskHighBg },
    medium: { fg: colors.riskMedium, bg: colors.riskMediumBg },
    low: { fg: colors.riskLow, bg: colors.riskLowBg },
  }[tier]
}
