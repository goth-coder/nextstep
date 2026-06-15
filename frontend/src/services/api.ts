import axios from 'axios'
import type { DriftInfo, ModelInfo, PedagogicalAdvice, PredictInput, PredictResult, StudentDetail, StudentSummary } from '../types/student'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '',
  timeout: 10_000,
})

// Throw on non-2xx so callers can render <ErrorState>
api.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(error)
)

// ── Student list ──────────────────────────────────────────────────────────────

export async function getStudents(): Promise<{ students: StudentSummary[]; total: number }> {
  // Longer timeout: cold start can take 60-90s (GCS download + model load from MLflow)
  const response = await api.get<{ students: StudentSummary[]; total: number }>('/api/students', { timeout: 30_000 })
  return response.data
}

// ── Student detail ────────────────────────────────────────────────────────────

export async function getStudent(id: number): Promise<StudentDetail> {
  // Longer timeout: cold start can take 60-90s (same as the list endpoint)
  const response = await api.get<StudentDetail>(`/api/students/${id}`, { timeout: 30_000 })
  return response.data
}

// ── AI advice ─────────────────────────────────────────────────────────────────

export async function getAdvice(id: number): Promise<PedagogicalAdvice> {
  const response = await api.get<PedagogicalAdvice>(`/api/students/${id}/advice`)
  return response.data
}

// ── Model info ────────────────────────────────────────────────────────────────

export async function getModelInfo(): Promise<ModelInfo> {
  const response = await api.get<ModelInfo>('/api/model')
  return response.data
}

// ── Model drift monitoring ────────────────────────────────────────────────────

export async function getModelDrift(): Promise<DriftInfo> {
  const response = await api.get<DriftInfo>('/api/model/drift')
  return response.data
}

// ── On-demand prediction ──────────────────────────────────────────────────────

export async function predict(input: PredictInput): Promise<PredictResult> {
  const response = await api.post<PredictResult>('/api/predict', input)
  return response.data
}

// ── Batch prediction ──────────────────────────────────────────────────────────

export interface BatchPredictItem extends PredictInput {
  student_id: number
}

export interface BatchPredictResult {
  results: Array<{
    student_id: number
    risk_score: number
    risk_tier: 'high' | 'medium' | 'low'
  }>
}

export async function predictBatch(students: BatchPredictItem[]): Promise<BatchPredictResult> {
  const response = await api.post<BatchPredictResult>('/api/predict/batch', { students }, {
    timeout: 30_000,   // scoring 1000+ students can take a few seconds
  })
  return response.data
}
