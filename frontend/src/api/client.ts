import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 60_000,
})

// ── Types ──────────────────────────────────────────────────────────────────

export interface Source {
  chunk_id: number | null
  paper_title: string
  page_number: number | null
  score: number
  chunk_text: string
}

export interface ChatRequest {
  message: string
  conversation_id?: string
  retrieval_strategy: 'dense' | 'hybrid' | 'hybrid_rerank'
  embedding_model: string
}

export interface ChatResponse {
  conversation_id: string
  answer: string
  sources: Source[]
  retrieval_strategy: string
  embedding_model: string
}

export interface Paper {
  id: number
  title: string
  authors: string | null
  source_filename: string
  arxiv_id: string | null
  total_pages: number | null
  uploaded_at: string
  chunk_count: number
}

export interface PaperUploadResponse {
  paper_id: number
  title: string
  message: string
  chunks_created: number
}

export interface HealthResponse {
  status: string
  db: string
  version: string
}

// ── API Functions ──────────────────────────────────────────────────────────

export const chatApi = {
  send: (req: ChatRequest) =>
    api.post<ChatResponse>('/chat', req).then((r) => r.data),
}

export const papersApi = {
  list: () => api.get<Paper[]>('/papers').then((r) => r.data),

  upload: (file: File, chunkingStrategy = 'recursive') => {
    const form = new FormData()
    form.append('file', file)
    return api
      .post<PaperUploadResponse>(
        `/papers/upload?chunking_strategy=${chunkingStrategy}`,
        form,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      .then((r) => r.data)
  },

  delete: (id: number) => api.delete(`/papers/${id}`),
}

export const healthApi = {
  check: () => api.get<HealthResponse>('/health').then((r) => r.data),
}
