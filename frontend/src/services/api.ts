import axios from 'axios'
import type { DriftInfo, ModelInfo, PedagogicalAdvice, StudentDetail, StudentSummary } from '../types/student'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080',
  timeout: 10_000,
})

// Throw on non-2xx so callers can render <ErrorState>
api.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(error)
)

// ── Student list ──────────────────────────────────────────────────────────────

export async function getStudents(): Promise<{ students: StudentSummary[]; total: number }> {
  const response = await api.get<{ students: StudentSummary[]; total: number }>('/api/students')
  return response.data
}

// ── Student detail ────────────────────────────────────────────────────────────

export async function getStudent(id: number): Promise<StudentDetail> {
  const response = await api.get<StudentDetail>(`/api/students/${id}`)
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
