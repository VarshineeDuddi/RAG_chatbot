import axios from 'axios'

const BASE = '/api'

export async function uploadFile(file, apiKey, onProgress) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('api_key', apiKey || '')

  const { data } = await axios.post(`${BASE}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      const pct = Math.round((e.loaded * 100) / e.total)
      onProgress?.(pct, `Uploading… ${pct}%`)
    },
  })
  return data
}

export async function listDocuments() {
  const { data } = await axios.get(`${BASE}/documents`)
  return data.documents
}

export async function deleteDocument(docId) {
  const { data } = await axios.delete(`${BASE}/documents/${docId}`)
  return data
}

/**
 * @param {string} question
 * @param {string} mode     - "api" or "local"
 * @param {string} apiKey   - Groq API key (only needed for "api" mode)
 * @param {string|null} docId
 */
export async function sendChat(question, mode, apiKey, docId) {
  const { data } = await axios.post(`${BASE}/chat`, {
    question,
    mode,
    api_key: apiKey,
    doc_id:  docId,
  })
  return data
}

export async function resetDB() {
  const { data } = await axios.post(`${BASE}/reset`)
  return data
}

export async function healthCheck() {
  const { data } = await axios.get(`${BASE}/health`)
  return data
}
