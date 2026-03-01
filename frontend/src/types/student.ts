// ── Risk tier ─────────────────────────────────────────────────────────────────
export type RiskTier = 'high' | 'medium' | 'low'

// ── Risk thresholds (mirrors backend: RISK_HIGH=0.7, RISK_MEDIUM=0.3) ─────────
export const RISK_HIGH = 0.7
export const RISK_MEDIUM = 0.3

// ── API response types ────────────────────────────────────────────────────────

export interface StudentSummary {
  student_id: number
  display_name: string
  phase: string
  class_group: string
  risk_score: number
  risk_tier: RiskTier
}

export interface ModelInfo {
  model_name: string
  version: string
  stage: string
  run_id: string
  run_name: string
  created_at: number       // unix ms
  trained_at: string       // ISO UTC
  source_script: string
  model_size_bytes: number | null
  pytorch_version: string
  python_version: string
  params: Record<string, string>
  metrics: Record<string, number>
}

export interface Indicators {
  iaa: number | null
  ieg: number | null
  ips: number | null
  ida: number | null
  ipv: number | null
  ipp: number | null
  inde: number | null
  defasagem: number | null
}

export interface StudentDetail extends StudentSummary {
  class_group: string
  fase_num: number | null
  gender: number | null   // 0 = Feminino, 1 = Masculino
  age: number | null
  indicators: Indicators
}

export interface PedagogicalAdvice {
  student_id: number
  advice: string
  is_fallback: boolean
  generated_at: string | null
}

export interface DriftInfo {
  total_students: number
  score_mean: number
  score_std: number
  score_p10: number
  score_p25: number
  score_p50: number
  score_p75: number
  score_p90: number
  tier_counts: { high: number; medium: number; low: number }
  histogram: Array<{ bucket: string; count: number; from: number; to: number }>
  computed_at: string
}
