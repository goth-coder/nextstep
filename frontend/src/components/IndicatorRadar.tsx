/**
 * SVG radar chart for student indicators — no external charting library required.
 * Renders up to 8 axes evenly distributed around a center point.
 */

import { colors, typography } from '../styles/theme'

interface IndicatorRadarProps {
  indicators: Record<string, number | null>
  labels: Record<string, string>
}

const SIZE = 240
const CENTER = SIZE / 2
const MAX_RADIUS = 90
const LEVELS = 4

function polarToXY(angle: number, radius: number) {
  // angle 0 = top, increases clockwise
  const rad = (angle - 90) * (Math.PI / 180)
  return {
    x: CENTER + radius * Math.cos(rad),
    y: CENTER + radius * Math.sin(rad),
  }
}

function pathFromPoints(points: { x: number; y: number }[]) {
  return (
    points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ') + ' Z'
  )
}

export default function IndicatorRadar({ indicators, labels }: IndicatorRadarProps) {
  const keys = Object.keys(labels).filter((k) => k in indicators)
  const n = keys.length
  if (n < 3) return null

  const angles = keys.map((_, i) => (360 / n) * i)

  // Web indicator values — normalize to 0-1.
  // indicators use 0–10 scale for most, and defasagem can be negative.
  // Clamp and normalize.
  function normalize(key: string, raw: number | null): number {
    if (raw === null || raw === undefined) return 0
    if (key === 'defasagem') {
      // defasagem: typically -8..+8; map -8→0, 0→0.5, +8→1 (higher lag = more risk = lower score)
      const clamped = Math.max(-8, Math.min(8, raw))
      return (8 - clamped) / 16  // positive defasagem = lower score on radar
    }
    // All others: 0..10 range, clamp
    return Math.max(0, Math.min(10, raw)) / 10
  }

  const values = keys.map((key) => normalize(key, indicators[key] ?? null))

  // Build grid rings
  const rings = Array.from({ length: LEVELS }, (_, i) => {
    const r = (MAX_RADIUS * (i + 1)) / LEVELS
    return keys.map((_, j) => polarToXY(angles[j], r))
  })

  // Build data polygon
  const dataPoints = keys.map((_k, i) => {
    const r = values[i] * MAX_RADIUS
    return polarToXY(angles[i], r)
  })

  return (
    <figure
      aria-label="Radar chart of student indicators"
      style={{ margin: 0, display: 'flex', flexDirection: 'column', alignItems: 'center' }}
    >
      <svg
        width={SIZE}
        height={SIZE}
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        style={{ overflow: 'visible' }}
        role="img"
        aria-label="Indicator radar"
      >
        {/* Grid rings */}
        {rings.map((points, ri) => (
          <polygon
            key={ri}
            points={points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')}
            fill="none"
            stroke={colors.gray200}
            strokeWidth="1"
          />
        ))}

        {/* Axis lines */}
        {keys.map((_, i) => {
          const outer = polarToXY(angles[i], MAX_RADIUS)
          return (
            <line
              key={i}
              x1={CENTER}
              y1={CENTER}
              x2={outer.x.toFixed(1)}
              y2={outer.y.toFixed(1)}
              stroke={colors.gray200}
              strokeWidth="1"
            />
          )
        })}

        {/* Data polygon fill */}
        <path
          d={pathFromPoints(dataPoints)}
          fill={`${colors.brandAccent}30`}
          stroke={colors.brandAccent}
          strokeWidth="2"
          strokeLinejoin="round"
        />

        {/* Data points */}
        {dataPoints.map((p, i) => (
          <circle
            key={i}
            cx={p.x.toFixed(1)}
            cy={p.y.toFixed(1)}
            r="4"
            fill={colors.brandAccent}
          />
        ))}

        {/* Axis labels */}
        {keys.map((k, i) => {
          const labelRadius = MAX_RADIUS + 22
          const pos = polarToXY(angles[i], labelRadius)
          const shortLabel = labels[k].split(' — ')[0].split(' – ')[0]
          return (
            <text
              key={k}
              x={pos.x.toFixed(1)}
              y={pos.y.toFixed(1)}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize="10"
              fontFamily={typography.fontFamily}
              fill={colors.gray600}
              fontWeight="600"
            >
              {shortLabel}
            </text>
          )
        })}
      </svg>
    </figure>
  )
}
