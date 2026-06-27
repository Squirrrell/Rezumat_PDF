import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { fetchMetrics, fetchPreview, askQuestion } from '../../api/client';
import { useDocument } from '../../context/DocumentContext';
import { useSettings } from '../../hooks/useSettings';
import ErrorBanner from '../common/ErrorBanner';
import LoadingSpinner from '../common/LoadingSpinner';
import Button from '../ui/Button';
import Card from '../ui/Card';
import Select from '../ui/Select';
import Slider from '../ui/Slider';

export default function SummaryTab() {
  const doc = useDocument();
  const settings = useSettings();
  const [question, setQuestion] = useState('');
  const [askedQuestion, setAskedQuestion] = useState('');
  const [localError, setLocalError] = useState(null);
  const [qaBusy, setQaBusy] = useState(false);

  useEffect(() => {
    if (!doc.documentId) return;
    fetchPreview(doc.documentId)
      .then(doc.setPreview)
      .catch(() => doc.setPreview(null));
    fetchMetrics(doc.documentId)
      .then((data) => doc.setDocumentMetrics(data.document_metrics || null))
      .catch(() => doc.setDocumentMetrics(null));
    doc.loadSummary();
    doc.loadComprehensiveSummary();
  }, [doc.documentId]);

  const onSummarize = async (source) => {
    if (!doc.documentId) return;
    setLocalError(null);
    try {
      await doc.summarize(settings.summaryPayload(source));
    } catch (err) {
      setLocalError(err.response?.data?.detail || err.message);
    }
  };

  const onSummarizeComprehensive = async (source) => {
    if (!doc.documentId) return;
    setLocalError(null);
    try {
      const data = await doc.summarizeComprehensive(settings.comprehensivePayload(source));
      if (data.instruct_warning) setLocalError(data.instruct_warning);
    } catch (err) {
      setLocalError(err.response?.data?.detail || err.message);
    }
  };

  const onQa = async (e) => {
    e?.preventDefault();
    if (!doc.documentId || !question.trim()) return;
    setQaBusy(true);
    setLocalError(null);
    setAskedQuestion(question);
    try {
      const data = await askQuestion(doc.documentId, settings.qaPayload(question));
      doc.setQaAnswer(data.answer);
      doc.setQaSources(data.sources);
      doc.setQaMetrics(data.metrics);
      setQuestion('');
      if (data.instruct_warning) setLocalError(data.instruct_warning);
    } catch (err) {
      setLocalError(err.response?.data?.detail || err.message);
    } finally {
      setQaBusy(false);
    }
  };

  if (!doc.documentId) {
    return (
      <p className="text-sm text-[var(--text-muted)]">
        Upload a PDF to get started.
      </p>
    );
  }

  return (
    <div className="space-y-8">
      <ErrorBanner message={localError || doc.error} onDismiss={() => setLocalError(null)} />

      <section>
        <h3 className="section-title mb-4">Brief summary</h3>
        <p className="mb-4 text-sm text-[var(--text-muted)]">
          Quick overview in a few sentences. Good when you want the main ideas fast.
        </p>

        <Card className="mb-4 flex flex-wrap items-end gap-6">
          <Select
            label="Summary length"
            value={settings.summaryLength}
            onChange={(e) => settings.setSummaryLength(e.target.value)}
            className="min-w-[10rem]"
          >
            <option value="short">Short</option>
            <option value="medium">Medium</option>
            <option value="long">Long</option>
          </Select>
          <Slider
            label="Max chunks"
            min={3}
            max={20}
            step={1}
            value={settings.summaryMaxChunks}
            onChange={settings.setSummaryMaxChunks}
            className="min-w-[12rem] flex-1"
          />
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => onSummarize('pdf')} disabled={doc.summarizing}>
              {doc.summarizing ? 'Summarizing...' : 'Summarize from PDF'}
            </Button>
            <Button
              variant="ghost"
              onClick={() => onSummarize('markdown')}
              disabled={doc.summarizing || !doc.hasMarkdown}
            >
              Summarize from Markdown
            </Button>
          </div>
        </Card>

        {doc.summarizing && <LoadingSpinner label="Generating summary..." />}

        {doc.summary && (
          <Card>
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <p className="text-sm font-semibold text-[var(--text)]">
                Brief summary {doc.summarySource ? `(from ${doc.summarySource})` : ''}
              </p>
              {doc.summaryMetrics && (
                <p className="text-xs text-[var(--text-muted)]">
                  {doc.summaryMetrics.summary_words} words ·{' '}
                  {doc.summaryMetrics.compression_percent?.toFixed(1)}% reduction ·{' '}
                  {doc.summaryMetrics.runtime_seconds?.toFixed(2)}s
                </p>
              )}
            </div>
            <p className="whitespace-pre-wrap text-sm leading-relaxed">{doc.summary}</p>
          </Card>
        )}
      </section>

      <section className="border-t border-[var(--border)] pt-8">
        <h3 className="section-title mb-4">Comprehensive summary</h3>
        <p className="mb-4 text-sm text-[var(--text-muted)]">
          Structured detailed summary (~1 page) covering main ideas, method, results, limitations,
          and more. Takes a bit longer than the brief summary.
        </p>

        <div className="mb-4 flex flex-wrap gap-2">
          <Button
            onClick={() => onSummarizeComprehensive('pdf')}
            disabled={doc.summarizingComprehensive}
          >
            {doc.summarizingComprehensive ? 'Building...' : 'Comprehensive from PDF'}
          </Button>
          <Button
            variant="ghost"
            onClick={() => onSummarizeComprehensive('markdown')}
            disabled={doc.summarizingComprehensive || !doc.hasMarkdown}
          >
            Comprehensive from Markdown
          </Button>
        </div>

        {doc.summarizingComprehensive && (
          <LoadingSpinner label="Building comprehensive summary..." />
        )}

        {doc.comprehensiveSummary && (
          <Card>
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <p className="text-sm font-semibold text-[var(--text)]">
                Comprehensive summary
                {doc.comprehensiveSummarySource
                  ? ` (from ${doc.comprehensiveSummarySource})`
                  : ''}
              </p>
              {doc.comprehensiveSummaryMetrics && (
                <p className="text-xs text-[var(--text-muted)]">
                  {doc.comprehensiveSummaryMetrics.summary_words} words ·{' '}
                  {doc.comprehensiveSummaryMetrics.runtime_seconds?.toFixed(2)}s
                </p>
              )}
            </div>
            <div className="markdown max-h-[70vh] overflow-y-auto pr-1">
              <ReactMarkdown>{doc.comprehensiveSummary}</ReactMarkdown>
            </div>
          </Card>
        )}
      </section>

      <section className="border-t border-[var(--border)] pt-8">
        <h3 className="section-title mb-4">Ask about the paper</h3>

        {doc.qaAnswer && (
          <div className="mb-4 space-y-3">
            <div className="flex justify-end">
              <div className="max-w-[85%] rounded-[var(--radius-xl)] rounded-br-md bg-[var(--accent)] px-4 py-3 text-sm font-medium text-[var(--accent-text)]">
                {askedQuestion || 'Your question'}
              </div>
            </div>
            <Card className="max-w-[85%] rounded-bl-md p-4">
              <p className="whitespace-pre-wrap text-sm">{doc.qaAnswer}</p>
            </Card>
          </div>
        )}

        {doc.qaMetrics && (
          <p className="mb-3 text-xs text-[var(--text-muted)]">
            {doc.qaMetrics.answer_words} answer words · {doc.qaMetrics.num_sources} sources ·{' '}
            {doc.qaMetrics.runtime_seconds?.toFixed(2)}s
          </p>
        )}

        <form
          onSubmit={onQa}
          className="glass-panel flex items-end gap-2 rounded-[var(--radius-xl)] p-2"
        >
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                onQa();
              }
            }}
            placeholder="e.g. What method did the authors propose?"
            rows={1}
            className="input-field max-h-32 flex-1 resize-none border-transparent bg-transparent shadow-none focus:shadow-none"
          />
          <button
            type="submit"
            disabled={!doc.documentId || qaBusy || !question.trim()}
            className="btn-primary !h-10 !w-10 !p-0 shrink-0"
            title="Get answer"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14m0 0l-6-6m6 6l-6 6" />
            </svg>
          </button>
        </form>
        {qaBusy && <LoadingSpinner label="Working..." />}
      </section>
    </div>
  );
}
