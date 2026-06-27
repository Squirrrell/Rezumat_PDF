import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 600000,
});

export async function fetchHealth() {
  const { data } = await api.get('/health');
  return data;
}

export async function uploadDocument(file, chunkSize, overlap) {
  const form = new FormData();
  form.append('file', file);
  form.append('chunk_size', String(chunkSize));
  form.append('overlap', String(overlap));
  const { data } = await api.post('/documents', form);
  return data;
}

export async function convertToMarkdown(documentId) {
  const { data } = await api.post(`/documents/${documentId}/convert-markdown`);
  return data;
}

export async function fetchPreview(documentId) {
  const { data } = await api.get(`/documents/${documentId}/preview`);
  return data;
}

export async function fetchMarkdown(documentId) {
  const { data } = await api.get(`/documents/${documentId}/markdown`);
  return data;
}

export async function summarizeDocument(documentId, body) {
  const { data } = await api.post(`/documents/${documentId}/summarize`, body);
  return data;
}

export async function fetchSummary(documentId) {
  const { data } = await api.get(`/documents/${documentId}/summary`);
  return data;
}

export async function summarizeComprehensive(documentId, body) {
  const { data } = await api.post(`/documents/${documentId}/summarize/comprehensive`, body);
  return data;
}

export async function fetchComprehensiveSummary(documentId) {
  const { data } = await api.get(`/documents/${documentId}/summary/comprehensive`);
  return data;
}

export async function askQuestion(documentId, qaBody) {
  const { data } = await api.post(`/documents/${documentId}/qa`, qaBody);
  return data;
}

export function exportMarkdownUrl(documentId) {
  return `/api/documents/${documentId}/export`;
}

export async function generateTestCards(documentId, body) {
  const { data } = await api.post(`/documents/${documentId}/test-cards/generate`, body);
  return data;
}

export async function generateTestCardAnswer(documentId, body) {
  const { data } = await api.post(`/documents/${documentId}/test-cards/answer`, body);
  return data;
}

export async function verifyTestCards(documentId, body) {
  const { data } = await api.post(`/documents/${documentId}/test-cards/verify`, body);
  return data;
}

export async function fetchMetrics(documentId) {
  const { data } = await api.get(`/documents/${documentId}/metrics`);
  return data;
}

export async function fetchTrainingInfo() {
  const { data } = await api.get('/training/info');
  return data;
}

export async function startTraining(body) {
  const { data } = await api.post('/training/start', body);
  return data;
}

export async function fetchTrainingJob(jobId) {
  const { data } = await api.get(`/training/jobs/${jobId}`);
  return data;
}

export async function cancelTraining(jobId) {
  const { data } = await api.post(`/training/jobs/${jobId}/cancel`);
  return data;
}

export async function evaluateTraining(body) {
  const { data } = await api.post('/training/evaluate', body);
  return data;
}

export default api;
