import axios from 'axios'

const http = axios.create({ baseURL: '/api' })

// Attach Bearer token if available
http.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle 401 globally
http.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err?.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.hash = '#/login'
    }
    return Promise.reject(err)
  },
)

// ── Knowledge Base API ──

export function listKBs() {
  return http.get('/knowledge-bases').then(r => r.data)
}

export function createKB(kbId) {
  return http.post('/knowledge-bases', null, { params: { kb_id: kbId } }).then(r => r.data)
}

export function deleteKB(kbId) {
  return http.delete(`/knowledge-bases/${kbId}`).then(r => r.data)
}

export function listDocs(kbId) {
  return http.get(`/knowledge-bases/${kbId}/documents`).then(r => r.data)
}

export async function uploadDoc(kbId, file) {
  const fd = new FormData()
  fd.append('file', file)
  try {
    const r = await http.post(`/knowledge-bases/${kbId}/documents`, fd)
    return r.data
  } catch (err) {
    // Extract server-side detail from 400/4xx responses
    const detail = err?.response?.data?.detail || err.message || 'Upload failed'
    throw new Error(detail)
  }
}

export function deleteDoc(kbId, filename) {
  return http.delete(`/knowledge-bases/${kbId}/documents/${encodeURIComponent(filename)}`).then(r => r.data)
}

// ── Chat API ──

export function sendChat(question, threadId, kbId) {
  return http.post('/chat', { question, thread_id: threadId, kb_id: kbId }).then(r => r.data)
}

export async function* streamChat(question, threadId, kbId) {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, thread_id: threadId, kb_id: kbId }),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(err.detail || 'Stream request failed')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const parts = buffer.split('\n\n')
    buffer = parts.pop() || ''

    for (const part of parts) {
      const lines = part.split('\n')
      let event = ''
      let data = ''
      for (const line of lines) {
        if (line.startsWith('event: ')) event = line.slice(7)
        if (line.startsWith('data: ')) data = line.slice(6)
      }
      if (data) {
        yield { event, data: JSON.parse(data) }
      }
    }
  }
}

// ── History API ──

export function listThreads(kbId) {
  const params = kbId ? { kb_id: kbId } : {}
  return http.get('/chat/history', { params }).then(r => r.data)
}

export function getThreadMessages(threadId) {
  return http.get(`/chat/history/${threadId}`).then(r => r.data)
}

export function deleteThread(threadId) {
  return http.delete(`/chat/history/${threadId}`).then(r => r.data)
}

// ── Auth API ──

export function register(data) {
  return http.post('/auth/register', data).then(r => r.data)
}

export function login(data) {
  return http.post('/auth/login', data).then(r => r.data)
}

export function sendCode(email) {
  return http.post('/auth/send-code', { email }).then(r => r.data)
}

export function loginWithCode(data) {
  return http.post('/auth/login-with-code', data).then(r => r.data)
}

export function getMe() {
  return http.get('/auth/me').then(r => r.data)
}

// ── Mind Map API ──

export function generateMindmap(threadId) {
  return http.post(`/chat/${threadId}/mindmap`).then(r => r.data)
}
