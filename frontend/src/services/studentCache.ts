/**
 * Module-level session cache for the student list.
 *
 * Lives for the lifetime of the JS bundle (page load → tab close).
 * Survives React router navigations — only evicted on full browser refresh,
 * which is exactly when we want fresh scores.
 *
 * Usage:
 *   import { getOrFetchStudents, invalidateStudentCache } from './studentCache'
 */

import type { StudentSummary } from '../types/student'
import { getStudents } from './api'

interface CacheEntry {
  students: StudentSummary[]
  total: number
  fetchedAt: number   // Date.now() ms
}

let _cache: CacheEntry | null = null
let _inflight: Promise<CacheEntry> | null = null

/**
 * Returns cached students instantly on subsequent calls.
 * Only one network request is ever in-flight at a time (deduped).
 * Retries with exponential backoff if API is not ready (503).
 */
export async function getOrFetchStudents(): Promise<CacheEntry> {
  if (_cache) return _cache

  // Deduplicate concurrent calls (e.g. StrictMode double-mount)
  if (_inflight) return _inflight

  _inflight = fetchWithRetry().then((data) => {
    _cache = { students: data.students, total: data.total, fetchedAt: Date.now() }
    _inflight = null
    return _cache
  }).catch((err) => {
    _inflight = null
    throw err
  })

  return _inflight
}

/**
 * Fetch students with exponential backoff retry for 503 (service not ready).
 * Max 5 attempts over ~15 seconds to handle API cold start.
 */
async function fetchWithRetry(attempt = 1): Promise<{ students: StudentSummary[]; total: number }> {
  const maxAttempts = 5
  const baseDelay = 1000 // ms

  try {
    return await getStudents()
  } catch (err: unknown) {
    const status = (err as { response?: { status?: number } })?.response?.status

    // Only retry on 503 (service not ready) - fail fast on other errors
    if (status === 503 && attempt < maxAttempts) {
      const delay = Math.min(baseDelay * Math.pow(2, attempt - 1), 8000)
      await new Promise(resolve => setTimeout(resolve, delay))
      return fetchWithRetry(attempt + 1)
    }

    throw err
  }
}

/** Call this to force a fresh fetch on the next getOrFetchStudents() call. */
export function invalidateStudentCache(): void {
  _cache = null
  _inflight = null
}

/** Age of the cache in seconds, or null if not loaded. */
export function cacheAgeSeconds(): number | null {
  return _cache ? Math.floor((Date.now() - _cache.fetchedAt) / 1000) : null
}
