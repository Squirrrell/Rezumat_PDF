import { createContext, useCallback, useContext, useMemo, useState } from 'react';

import {
  convertToMarkdown,
  fetchComprehensiveSummary,
  fetchMarkdown,
  fetchSummary,
  summarizeComprehensive,
  summarizeDocument,
  uploadDocument,
} from '../api/client';

const DocumentContext = createContext(null);

export function DocumentProvider({ children }) {
  const [documentId, setDocumentId] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [preview, setPreview] = useState(null);
  const [markdown, setMarkdown] = useState('');
  const [hasMarkdown, setHasMarkdown] = useState(false);
  const [summary, setSummary] = useState('');
  const [summaryMetrics, setSummaryMetrics] = useState(null);
  const [summarySource, setSummarySource] = useState(null);
  const [comprehensiveSummary, setComprehensiveSummary] = useState('');
  const [comprehensiveSummaryMetrics, setComprehensiveSummaryMetrics] = useState(null);
  const [comprehensiveSummarySource, setComprehensiveSummarySource] = useState(null);
  const [documentMetrics, setDocumentMetrics] = useState(null);
  const [qaAnswer, setQaAnswer] = useState('');
  const [qaSources, setQaSources] = useState([]);
  const [qaMetrics, setQaMetrics] = useState(null);
  const [testCards, setTestCards] = useState([]);
  const [testCardResults, setTestCardResults] = useState(null);
  const [lastRuntime, setLastRuntime] = useState({});
  const [loading, setLoading] = useState(false);
  const [convertingMarkdown, setConvertingMarkdown] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [summarizingComprehensive, setSummarizingComprehensive] = useState(false);
  const [error, setError] = useState(null);

  const resetInteractionResults = useCallback(() => {
    setQaAnswer('');
    setQaSources([]);
    setQaMetrics(null);
    setTestCards([]);
    setTestCardResults(null);
    setSummary('');
    setSummaryMetrics(null);
    setSummarySource(null);
    setComprehensiveSummary('');
    setComprehensiveSummaryMetrics(null);
    setComprehensiveSummarySource(null);
    setMarkdown('');
    setHasMarkdown(false);
  }, []);

  const applyMetadata = useCallback((data) => {
    setDocumentId(data.document_id);
    setMetadata(data);
    setLastRuntime(data.last_runtime || {});
    setHasMarkdown(Boolean(data.has_markdown));
    if (!data.has_markdown) {
      setMarkdown('');
    }
    if (!data.has_summary) {
      setSummary('');
      setSummaryMetrics(null);
      setSummarySource(null);
    }
  }, []);

  const uploadPdf = useCallback(
    async (file, chunkSize, overlap) => {
      setLoading(true);
      setError(null);
      try {
        const data = await uploadDocument(file, chunkSize, overlap);
        applyMetadata(data);
        resetInteractionResults();
        return data;
      } catch (err) {
        const msg = err.response?.data?.detail || err.message;
        setError(msg);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [applyMetadata, resetInteractionResults],
  );

  const runConvertToMarkdown = useCallback(async () => {
    if (!documentId) return null;
    setConvertingMarkdown(true);
    setError(null);
    try {
      const data = await convertToMarkdown(documentId);
      applyMetadata(data);
      const md = await fetchMarkdown(documentId);
      setMarkdown(md.markdown || '');
      setHasMarkdown(true);
      return data;
    } catch (err) {
      const msg = err.response?.data?.detail || err.message;
      setError(msg);
      throw err;
    } finally {
      setConvertingMarkdown(false);
    }
  }, [documentId, applyMetadata]);

  const runSummarize = useCallback(
    async (body) => {
      if (!documentId) return null;
      setSummarizing(true);
      setError(null);
      try {
        const data = await summarizeDocument(documentId, body);
        setSummary(data.summary || '');
        setSummaryMetrics(data.metrics || null);
        setSummarySource(data.source || body.source);
        setMetadata((prev) =>
          prev ? { ...prev, has_summary: true } : prev,
        );
        setLastRuntime((prev) => ({
          ...prev,
          summary_total: data.metrics?.runtime_seconds,
        }));
        return data;
      } catch (err) {
        const msg = err.response?.data?.detail || err.message;
        setError(msg);
        throw err;
      } finally {
        setSummarizing(false);
      }
    },
    [documentId],
  );

  const runSummarizeComprehensive = useCallback(
    async (body) => {
      if (!documentId) return null;
      setSummarizingComprehensive(true);
      setError(null);
      try {
        const data = await summarizeComprehensive(documentId, body);
        setComprehensiveSummary(data.summary || '');
        setComprehensiveSummaryMetrics(data.metrics || null);
        setComprehensiveSummarySource(data.source || body.source);
        setLastRuntime((prev) => ({
          ...prev,
          comprehensive_summary_total: data.metrics?.runtime_seconds,
        }));
        return data;
      } catch (err) {
        const msg = err.response?.data?.detail || err.message;
        setError(msg);
        throw err;
      } finally {
        setSummarizingComprehensive(false);
      }
    },
    [documentId],
  );

  const loadComprehensiveSummary = useCallback(async () => {
    if (!documentId) return;
    try {
      const data = await fetchComprehensiveSummary(documentId);
      setComprehensiveSummary(data.summary || '');
      setComprehensiveSummaryMetrics(data.metrics || null);
      setComprehensiveSummarySource(data.source || null);
    } catch {
      /* no comprehensive summary yet */
    }
  }, [documentId]);

  const loadSummary = useCallback(async () => {
    if (!documentId) return;
    try {
      const data = await fetchSummary(documentId);
      setSummary(data.summary || '');
      setSummaryMetrics(data.metrics || null);
      setSummarySource(data.source || null);
    } catch {
      /* no summary yet */
    }
  }, [documentId]);

  const loadMarkdown = useCallback(async () => {
    if (!documentId || !hasMarkdown) return;
    try {
      const data = await fetchMarkdown(documentId);
      setMarkdown(data.markdown || '');
    } catch {
      setMarkdown('');
    }
  }, [documentId, hasMarkdown]);

  const value = useMemo(
    () => ({
      documentId,
      metadata,
      preview,
      setPreview,
      markdown,
      setMarkdown,
      hasMarkdown,
      summary,
      setSummary,
      summaryMetrics,
      setSummaryMetrics,
      summarySource,
      comprehensiveSummary,
      setComprehensiveSummary,
      comprehensiveSummaryMetrics,
      setComprehensiveSummaryMetrics,
      comprehensiveSummarySource,
      documentMetrics,
      setDocumentMetrics,
      qaAnswer,
      setQaAnswer,
      qaSources,
      setQaSources,
      qaMetrics,
      setQaMetrics,
      testCards,
      setTestCards,
      testCardResults,
      setTestCardResults,
      lastRuntime,
      setLastRuntime,
      loading,
      convertingMarkdown,
      summarizing,
      summarizingComprehensive,
      error,
      setError,
      uploadPdf,
      convertToMarkdown: runConvertToMarkdown,
      summarize: runSummarize,
      summarizeComprehensive: runSummarizeComprehensive,
      loadSummary,
      loadComprehensiveSummary,
      loadMarkdown,
      resetInteractionResults,
    }),
    [
      documentId,
      metadata,
      preview,
      markdown,
      hasMarkdown,
      summary,
      summaryMetrics,
      summarySource,
      comprehensiveSummary,
      comprehensiveSummaryMetrics,
      comprehensiveSummarySource,
      documentMetrics,
      qaAnswer,
      qaSources,
      qaMetrics,
      testCards,
      testCardResults,
      lastRuntime,
      loading,
      convertingMarkdown,
      summarizing,
      summarizingComprehensive,
      error,
      uploadPdf,
      runConvertToMarkdown,
      runSummarize,
      runSummarizeComprehensive,
      loadSummary,
      loadComprehensiveSummary,
      loadMarkdown,
      resetInteractionResults,
    ],
  );

  return <DocumentContext.Provider value={value}>{children}</DocumentContext.Provider>;
}

export function useDocument() {
  const ctx = useContext(DocumentContext);
  if (!ctx) throw new Error('useDocument must be used within DocumentProvider');
  return ctx;
}
