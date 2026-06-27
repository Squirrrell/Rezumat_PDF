import { useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { exportMarkdownUrl } from '../../api/client';
import { useDocument } from '../../context/DocumentContext';
import ErrorBanner from '../common/ErrorBanner';
import LoadingSpinner from '../common/LoadingSpinner';
import Button from '../ui/Button';
import Card from '../ui/Card';

export default function MarkdownTab() {
  const doc = useDocument();

  useEffect(() => {
    if (doc.documentId && doc.hasMarkdown) {
      doc.loadMarkdown();
    }
  }, [doc.documentId, doc.hasMarkdown]);

  if (!doc.documentId) {
    return (
      <p className="text-sm text-[var(--text-muted)]">
        Upload a PDF to get started. Markdown conversion is optional and runs when you need it.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      {doc.error && <ErrorBanner message={doc.error} />}

      <section>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="section-title">Markdown conversion</h3>
            <p className="mt-1 text-sm text-[var(--text-muted)]">
              Convert with Marker when you need formatted Markdown. This is separate from fast PDF
              indexing used for summary and Q&A.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => doc.convertToMarkdown()} disabled={doc.convertingMarkdown}>
              {doc.convertingMarkdown ? 'Converting...' : 'Convert to Markdown'}
            </Button>
            {doc.hasMarkdown && doc.documentId && (
              <a href={exportMarkdownUrl(doc.documentId)}>
                <Button variant="ghost">Export .md</Button>
              </a>
            )}
          </div>
        </div>

        {doc.convertingMarkdown && (
          <LoadingSpinner label="Converting PDF to Markdown with Marker..." />
        )}

        {doc.hasMarkdown && doc.markdown ? (
          <Card>
            <div className="markdown max-h-[70vh] overflow-y-auto pr-1">
              <ReactMarkdown>{doc.markdown}</ReactMarkdown>
            </div>
          </Card>
        ) : (
          !doc.convertingMarkdown && (
            <p className="text-sm text-[var(--text-muted)]">
              No Markdown yet. Click &quot;Convert to Markdown&quot; to run Marker on this PDF.
            </p>
          )
        )}
      </section>
    </div>
  );
}
