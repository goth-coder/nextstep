/**
 * SVG radar chart for student indicators — no external charting library required.
 * Renders up to 8 axes evenly distributed around a center point.
 * Data points show a numeric tooltip on hover.
 */

import { useState } from 'react'
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

interface TooltipState {
  idx: number
  x: number
  y: number
}

function formatRawValue(key: string, raw: number | null): string {
  if (raw === null || raw === undefined) return 'N/A'
  if (key === 'defasagem') {
    const v = Math.round(raw)
    return `${v > 0 ? '+' : ''}${v} ${Math.abs(v) === 1 ? 'fase' : 'fases'}`
  }
  return raw.toFixed(2)
}

export default function IndicatorRadar({ indicators, labels }: IndicatorRadarProps) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null)

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

        {/* Data points — hover to see numeric value */}
        {dataPoints.map((p, i) => (
          <g
            key={i}
            style={{ cursor: 'pointer' }}
            onMouseEnter={() => setTooltip({ idx: i, x: p.x, y: p.y })}
            onMouseLeave={() => setTooltip(null)}
          >
            {/* Larger invisible hit area */}
            <circle cx={p.x.toFixed(1)} cy={p.y.toFixed(1)} r="10" fill="transparent" />
            <circle
              cx={p.x.toFixed(1)}
              cy={p.y.toFixed(1)}
              r={tooltip?.idx === i ? '5.5' : '4'}
              fill={colors.brandAccent}
              stroke="white"
              strokeWidth={tooltip?.idx === i ? '2' : '0'}
              style={{ transition: 'r 0.1s, stroke-width 0.1s' }}
            />
          </g>
        ))}

        {/* Tooltip */}
        {tooltip !== null && (() => {
          const key = keys[tooltip.idx]
          const label = labels[key].split(' — ')[0].split(' – ')[0]
          const value = formatRawValue(key, indicators[key] ?? null)
          const text = `${label}: ${value}`
          const charWidth = 6.5
          const boxW = text.length * charWidth + 16
          const boxH = 22
          // Offset tooltip so it doesn't overlap the point; flip if too close to edge
          const offX = tooltip.x + boxW / 2 + 8 > SIZE ? -boxW / 2 - 8 : boxW / 2 + 8
          const offY = tooltip.y - boxH - 8 < 0 ? boxH + 8 : -boxH / 2
          const tx = tooltip.x + offX
          const ty = tooltip.y + offY
          return (
            <g style={{ pointerEvents: 'none' }}>
              <rect
                x={(tx - boxW / 2).toFixed(1)}
                y={(ty - boxH / 2).toFixed(1)}
                width={boxW.toFixed(1)}
                height={boxH.toFixed(1)}
                rx="5"
                fill={colors.gray900}
                opacity="0.92"
              />
              <text
                x={tx.toFixed(1)}
                y={ty.toFixed(1)}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize="11"
                fontFamily={typography.fontFamily}
                fontWeight="600"
                fill="white"
              >
                {text}
              </text>
            </g>
          )
        })()}

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
