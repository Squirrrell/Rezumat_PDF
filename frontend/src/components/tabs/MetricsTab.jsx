import { useEffect, useState } from 'react';
import { fetchHealth, fetchMetrics } from '../../api/client';
import { useDocument } from '../../context/DocumentContext';
import DocumentMetricsRow from '../DocumentMetricsRow';
import LoadingSpinner from '../common/LoadingSpinner';
import Card from '../ui/Card';

export default function MetricsTab() {
  const doc = useDocument();
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    if (!doc.documentId) {
      setMetrics(null);
      return;
    }
    setLoading(true);
    fetchMetrics(doc.documentId)
      .then(setMetrics)
      .finally(() => setLoading(false));
  }, [doc.documentId, doc.documentMetrics, doc.qaMetrics, doc.summaryMetrics]);

  if (!doc.documentId) {
    return <p className="text-sm text-[var(--text-muted)]">Upload a PDF to get started.</p>;
  }
  if (loading) return <LoadingSpinner label="Loading metrics..." />;

  return (
    <div className="space-y-8">
      <section>
        <h3 className="section-title mb-4">Document metrics</h3>
        {metrics?.document_metrics ? (
          <>
            <DocumentMetricsRow metrics={metrics.document_metrics} />
            <Card className="mt-4 overflow-hidden p-0">
              <table className="w-full border-collapse text-sm">
                <tbody>
                  {Object.entries(metrics.document_metrics).map(([k, v]) => (
                    <tr
                      key={k}
                      className="border-t border-[var(--border)] first:border-t-0 transition-colors hover:bg-[var(--surface-hover)]"
                    >
                      <td className="p-3 font-medium">{k}</td>
                      <td className="p-3 text-[var(--text-muted)]">{String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </>
        ) : (
          <p className="text-sm text-[var(--text-muted)]">
            Upload a PDF to see ingest metrics.
          </p>
        )}
      </section>

      <section>
        <h3 className="section-title mb-4">Summary metrics</h3>
        {metrics?.summary_metrics ? (
          <Card>
            <pre className="overflow-x-auto text-xs text-[var(--text-muted)]">
              {JSON.stringify(metrics.summary_metrics, null, 2)}
            </pre>
          </Card>
        ) : (
          <p className="text-sm text-[var(--text-muted)]">
            Run summarization on the Summary tab to see summary metrics.
          </p>
        )}
      </section>

      <section>
        <h3 className="section-title mb-4">Q&A metrics</h3>
        {metrics?.qa_metrics ? (
          <Card>
            <pre className="overflow-x-auto text-xs text-[var(--text-muted)]">
              {JSON.stringify(metrics.qa_metrics, null, 2)}
            </pre>
          </Card>
        ) : (
          <p className="text-sm text-[var(--text-muted)]">Ask a question to see Q&A metrics.</p>
        )}
      </section>

      <section>
        <h3 className="section-title mb-4">All runtimes</h3>
        {Object.keys(metrics?.last_runtime || {}).length ? (
          <ul className="space-y-2">
            {Object.entries(metrics.last_runtime).map(([label, seconds]) => (
              <li key={label}>
                <Card className="flex justify-between p-3 text-sm">
                  <span className="text-[var(--text-muted)]">{label}</span>
                  <span className="font-mono font-medium">{Number(seconds).toFixed(2)}s</span>
                </Card>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-[var(--text-muted)]">No timing data yet.</p>
        )}
      </section>

      {health && (
        <p className="text-xs text-[var(--text-muted)]">
          Device: {health.device?.toUpperCase()}
        </p>
      )}
    </div>
  );
}
