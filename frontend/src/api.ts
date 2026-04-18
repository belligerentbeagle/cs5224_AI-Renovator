/**
 * API client for RoomStyle backend.
 *
 * Local dev  → http://localhost:8000       (uvicorn)
 * Production → API Gateway URL             (Lambda + Mangum)
 *
 * Set VITE_API_URL in .env for the target environment.
 */
const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function request<T>(path: string, init?: RequestInit, token?: string): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(init?.headers as Record<string, string> | undefined),
  }
  const res = await fetch(`${BASE}${path}`, { ...init, headers })
  if (!res.ok) {
    const bodyText = await res.text().catch(() => '')
    let detail = bodyText
    try {
      const parsed = JSON.parse(bodyText) as { detail?: string }
      detail = parsed.detail || bodyText
    } catch {
      // Keep the raw body text when the response isn't JSON.
    }
    throw new Error(detail || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export const login = (email: string, password: string) =>
  request<{ user_id: string; token: string }>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  })

export const register = (email: string, password: string, name?: string) =>
  request<{ user_id: string; token: string }>('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ email, password, name }),
  })

// ── Products ──────────────────────────────────────────────────────────────────

export const searchProducts = (params: {
  q?: string
  style?: string
  min_price?: number
  max_price?: number
  source?: string
}) => {
  const qs = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => v !== undefined && qs.set(k, String(v)))
  return request<import('./types').Product[]>(`/products?${qs}`)
}

export const indexFromUrl = (url: string) =>
  request<import('./types').Product>('/products/from-url', {
    method: 'POST',
    body: JSON.stringify({ url }),
  })

// ── Projects ──────────────────────────────────────────────────────────────────

export const createProject = (
  token: string,
  data: { title: string; style_prompt?: string; budget_limit?: number },
) =>
  request<import('./types').Project>('/projects', {
    method: 'POST',
    body: JSON.stringify(data),
  }, token)

export const listProjects = (token: string) =>
  request<import('./types').Project[]>('/projects', undefined, token)

export const presignUpload = (
  token: string,
  projectId: string,
  file_name: string,
  content_type: string,
) =>
  request<{ photo_id: string; upload_url: string; s3_key: string; expires_in: number }>(
    `/projects/${projectId}/uploads/presign`,
    { method: 'POST', body: JSON.stringify({ file_name, content_type }) },
    token,
  )

export const uploadFileToS3 = (uploadUrl: string, file: File) =>
  fetch(uploadUrl, { method: 'PUT', body: file, headers: { 'Content-Type': file.type } })

// ── Generation ────────────────────────────────────────────────────────────────

type FurniturePayload = {
  name: string
  image_url?: string
  product_id?: string
  price: number
  source: 'ikea' | 'taobao' | 'scraped'
  buy_url?: string
}

export const generateRoom = (
  token: string,
  project_id: string,
  photo_id: string,
  furniture: FurniturePayload[],
  style_name: string,
  prompt_text?: string,
) =>
  request<import('./types').GenerationPending>('/generate/room', {
    method: 'POST',
    body: JSON.stringify({ project_id, photo_id, furniture, style_name, prompt_text }),
  }, token)

export const designForMe = (
  token: string,
  project_id: string,
  photo_id: string | null,
  style_name: string,
  prompt_text?: string,
) =>
  request<import('./types').GenerationPending>('/generate/design-for-me', {
    method: 'POST',
    body: JSON.stringify({ project_id, photo_id, style_name, prompt_text }),
  }, token)

export const pollGeneration = (token: string, generationId: string) =>
  request<import('./types').GenerationResult>(`/generations/${generationId}`, undefined, token)

// ── Cart ──────────────────────────────────────────────────────────────────────

export const getCart = (token: string, projectId: string) =>
  request<import('./types').CartResponse>(
    `/projects/${projectId}/cart`,
    undefined,
    token,
  )
