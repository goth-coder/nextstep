import { useCallback, useEffect, useState } from 'react'
import AppShell from '../components/AppShell'
import ErrorState from '../components/ErrorState'
import { getModelInfo } from '../services/api'
import { colors, radius, shadows, typography } from '../styles/theme'
import type { ModelInfo } from '../types/student'

type Status = 'loading' | 'success' | 'error'

// Human-readable labels for params
const PARAM_LABELS: Record<string, string> = {
  hidden_size: 'Hidden size',
  num_layers: 'LSTM layers',
  epochs: 'Training epochs',
  lr: 'Learning rate',
  batch_size: 'Batch size',
  input_size: 'Input features',
  seed: 'Random seed',
  scaler: 'Feature scaler',
  split: 'Train/test split',
  val_split: 'Validation split',
  threshold_method: 'Threshold method',
}

const METRIC_LABELS: Record<string, { label: string; format: (v: number) => string; tip: string }> = {
  val_auc: {
    label: 'Val AUC-ROC',
    format: (v) => v.toFixed(4),
    tip: 'Area under ROC curve on the 2024 test set. Higher = better.',
  },
  val_f1: {
    label: 'Val F1',
    format: (v) => v.toFixed(4),
    tip: 'F1-score at the optimized threshold on the 2024 test set.',
  },
  val_f1_internal: {
    label: 'Val F1 (internal)',
    format: (v) => v.toFixed(4),
    tip: 'F1 on the last-20% validation split used for threshold selection.',
  },
  threshold: {
    label: 'Decision threshold',
    format: (v) => v.toFixed(4),
    tip: 'Probability cutoff chosen by PR-curve F1-max on the validation set.',
  },
  train_loss: {
    label: 'Train loss (BCE)',
    format: (v) => v.toFixed(6),
    tip: 'Final Binary Cross-Entropy loss on the training set.',
  },
}

const FEATURE_ORDER = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'INDE', 'Defasagem', 'Fase']

const FEATURE_DESCRIPTIONS: Record<string, string> = {
  IAA: 'Academic Performance Index',
  IEG: 'Engagement Index',
  IPS: 'Psychosocial Index',
  IDA: 'Self-sufficiency Index',
  IPV: 'Life Vision Index',
  INDE: 'Educational Development Index (composite)',
  Defasagem: 'School lag (years behind grade level)',
  Fase: 'Normalized phase number (0–8)',
}

function bytes(n: number | null) {
  if (n == null) return '—'
  if (n > 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(2)} MB`
  if (n > 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${n} B`
}

function msToDate(ms: number) {
  return new Date(ms).toLocaleString('en-GB', {
    dateStyle: 'long',
    timeStyle: 'short',
  })
}

export default function ModelPage() {
  const [status, setStatus] = useState<Status>('loading')
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [errorMsg, setErrorMsg] = useState('')

  const fetch = useCallback(async () => {
    setStatus('loading')
    try {
      setModel(await getModelInfo())
      setStatus('success')
    } catch {
      setErrorMsg('Could not load model information from MLflow.')
      setStatus('error')
    }
  }, [])

  useEffect(() => { fetch() }, [fetch])

  return (
    <AppShell>
      <div style={{ marginBottom: '1.5rem' }}>
        <h1 style={{
          fontSize: typography.sizes['2xl'], fontWeight: 800,
          color: colors.brandPrimary, margin: 0, letterSpacing: '-0.02em',
        }}>
          Active Model
        </h1>
        <p style={{ color: colors.gray500, marginTop: '0.25rem', fontSize: typography.sizes.sm }}>
          Artifact registered in MLflow — currently used for risk inference
        </p>
      </div>

      {status === 'loading' && (
        <div role="status" aria-live="polite"
          style={{ textAlign: 'center', padding: '5rem', color: colors.gray500 }}>
          <div style={{ fontSize: '2rem', marginBottom: '0.75rem' }}>⏳</div>
          Loading model info…
        </div>
      )}

      {status === 'error' && <ErrorState message={errorMsg} onRetry={fetch} />}

      {status === 'success' && model && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', maxWidth: '900px' }}>

          {/* ── Identity card ─────────────────────────────────────── */}
          <Card>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                  <h2 style={{ fontSize: typography.sizes.xl, fontWeight: 800, color: colors.brandPrimary, margin: 0 }}>
                    {model.model_name}
                  </h2>
                  <span style={{
                    fontSize: typography.sizes.xs, fontWeight: 700,
                    background: colors.brandAccent, color: colors.white,
                    borderRadius: radius.full, padding: '0.2rem 0.7rem',
                    letterSpacing: '0.04em',
                  }}>
                    v{model.version}
                  </span>
                  {model.stage && model.stage !== 'None' && (
                    <span style={{
                      fontSize: typography.sizes.xs, fontWeight: 700,
                      background: colors.riskLowBg, color: colors.riskLow,
                      borderRadius: radius.full, padding: '0.2rem 0.7rem',
                      letterSpacing: '0.04em',
                    }}>
                      {model.stage}
                    </span>
                  )}
                </div>
                <p style={{ color: colors.gray500, fontSize: typography.sizes.sm, marginTop: '0.4rem', marginBottom: 0 }}>
                  Run: <code style={{ fontFamily: 'monospace', color: colors.gray700 }}>{model.run_id.slice(0, 8)}</code>
                  {model.run_name && <> · <em>{model.run_name}</em></>}
                </p>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: typography.sizes.xs, color: colors.gray500 }}>Artifact size</div>
                <div style={{ fontSize: typography.sizes.xl, fontWeight: 800, color: colors.brandPrimary }}>
                  {bytes(model.model_size_bytes)}
                </div>
              </div>
            </div>

            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
              gap: '0.5rem 2rem', marginTop: '1rem',
              paddingTop: '1rem', borderTop: `1px solid ${colors.gray100}`,
            }}>
              <MetaRow label="Trained at" value={model.trained_at ? new Date(model.trained_at).toLocaleString('en-GB') : '—'} />
              <MetaRow label="Registered at" value={model.created_at ? msToDate(model.created_at) : '—'} />
              <MetaRow label="Source script" value={model.source_script || '—'} mono />
              <MetaRow label="PyTorch version" value={model.pytorch_version || '—'} mono />
              <MetaRow label="Python version" value={model.python_version || '—'} mono />
            </div>
          </Card>

          {/* ── Metrics ──────────────────────────────────────────────── */}
          <Section title="Evaluation Metrics">
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
              gap: '0.75rem',
            }}>
              {Object.entries(METRIC_LABELS).map(([key, meta]) => {
                const val = model.metrics[key]
                if (val === undefined) return null
                const isGood = key === 'val_auc' || key === 'val_f1'
                const accent = isGood
                  ? (val >= 0.75 ? colors.riskLow : val >= 0.6 ? colors.riskMedium : colors.riskHigh)
                  : colors.brandPrimary
                return (
                  <div key={key} style={{
                    background: colors.white, border: `1px solid ${colors.gray200}`,
                    borderRadius: radius.md, padding: '0.875rem 1rem',
                    boxShadow: shadows.sm,
                  }}
                    title={meta.tip}
                  >
                    <div style={{ fontSize: typography.sizes.xs, fontWeight: 700, color: colors.gray500, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.3rem' }}>
                      {meta.label}
                    </div>
                    <div style={{ fontSize: typography.sizes['2xl'], fontWeight: 800, color: accent, fontVariantNumeric: 'tabular-nums' }}>
                      {meta.format(val)}
                    </div>
                  </div>
                )
              })}
            </div>
            <p style={{ fontSize: typography.sizes.xs, color: colors.gray500, marginTop: '0.75rem' }}>
              ℹ️ Hover each metric card for a description. Green = good (AUC ≥ 0.75, F1 ≥ 0.75), yellow = acceptable, red = watch.
            </p>
          </Section>

          {/* ── Hyperparameters ───────────────────────────────────────── */}
          <Section title="Hyperparameters">
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
              gap: '0.5rem',
            }}>
              {Object.entries(model.params).map(([key, val]) => (
                <div key={key} style={{
                  display: 'flex', flexDirection: 'column', gap: '0.25rem',
                  padding: '0.625rem 0.875rem',
                  background: colors.gray50, borderRadius: radius.sm,
                  border: `1px solid ${colors.gray200}`,
                }}>
                  <span style={{ fontSize: typography.sizes.xs, color: colors.gray500, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    {PARAM_LABELS[key] ?? key}
                  </span>
                  <code style={{
                    fontSize: typography.sizes.sm, fontFamily: 'monospace',
                    color: colors.brandPrimary, fontWeight: 700,
                    wordBreak: 'break-all',
                  }}>
                    {val}
                  </code>
                </div>
              ))}
            </div>
          </Section>

          {/* ── Architecture ─────────────────────────────────────────── */}
          <Section title="Model Architecture">
            <div style={{
              background: colors.gray50, border: `1px solid ${colors.gray200}`,
              borderRadius: radius.md, padding: '1rem 1.25rem',
              fontFamily: 'monospace', fontSize: typography.sizes.sm,
              color: colors.gray700, lineHeight: 1.7,
            }}>
              <div><span style={{ color: colors.brandAccent, fontWeight: 700 }}>Type</span>: LSTM (Long Short-Term Memory)</div>
              <div><span style={{ color: colors.brandAccent, fontWeight: 700 }}>Input features</span> ({model.params.input_size}): IAA · IEG · IPS · IDA · IPV · INDE · Defasagem · Fase</div>
              <div><span style={{ color: colors.brandAccent, fontWeight: 700 }}>LSTM hidden size</span>: {model.params.hidden_size}</div>
              <div><span style={{ color: colors.brandAccent, fontWeight: 700 }}>LSTM layers</span>: {model.params.num_layers}</div>
              <div><span style={{ color: colors.brandAccent, fontWeight: 700 }}>Output</span>: sigmoid → P(dropout risk) ∈ [0, 1]</div>
              <div><span style={{ color: colors.brandAccent, fontWeight: 700 }}>Threshold</span>: {model.metrics.threshold?.toFixed(4)} (PR-curve F1-max on val set)</div>
              <div><span style={{ color: colors.brandAccent, fontWeight: 700 }}>Scaler</span>: {model.params.scaler} (fit on train only, clip ±5 σ)</div>
            </div>
          </Section>

          {/* ── Input features ───────────────────────────────────────── */}
          <Section title="Input Features">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {FEATURE_ORDER.map((feat, i) => (
                <div key={feat} style={{
                  display: 'flex', alignItems: 'center', gap: '0.75rem',
                  padding: '0.5rem 0.875rem',
                  background: i % 2 === 0 ? colors.white : colors.gray50,
                  borderRadius: radius.sm, border: `1px solid ${colors.gray200}`,
                }}>
                  <span style={{
                    minWidth: '2.5rem', fontSize: typography.sizes.xs,
                    fontWeight: 800, color: colors.brandAccent, fontFamily: 'monospace',
                  }}>
                    F{i + 1}
                  </span>
                  <span style={{ fontWeight: 700, color: colors.brandPrimary, minWidth: '80px', fontSize: typography.sizes.sm }}>
                    {feat}
                  </span>
                  <span style={{ color: colors.gray600, fontSize: typography.sizes.sm }}>
                    {FEATURE_DESCRIPTIONS[feat]}
                  </span>
                </div>
              ))}
            </div>
          </Section>

        </div>
      )}
    </AppShell>
  )
}

// ── Shared sub-components ─────────────────────────────────────────────────────

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      background: colors.white, border: `1px solid ${colors.gray200}`,
      borderRadius: radius.xl, padding: '1.5rem 2rem', boxShadow: shadows.md,
    }}>
      {children}
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: colors.white, border: `1px solid ${colors.gray200}`,
      borderRadius: radius.lg, padding: '1.25rem 1.5rem', boxShadow: shadows.sm,
    }}>
      <h2 style={{
        fontSize: typography.sizes.xs, fontWeight: 700,
        color: colors.gray500, textTransform: 'uppercase',
        letterSpacing: '0.07em', margin: '0 0 1rem 0',
      }}>
        {title}
      </h2>
      {children}
    </div>
  )
}

function MetaRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.1rem' }}>
      <span style={{ fontSize: typography.sizes.xs, color: colors.gray500, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
        {label}
      </span>
      <span style={{ fontSize: typography.sizes.sm, color: colors.gray700, fontFamily: mono ? 'monospace' : 'inherit', fontWeight: 500 }}>
        {value}
      </span>
    </div>
  )
}
