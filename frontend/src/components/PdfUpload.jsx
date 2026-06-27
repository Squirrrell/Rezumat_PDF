import { useRef, useState } from 'react';
import { useDocument } from '../context/DocumentContext';
import { useSettings } from '../hooks/useSettings';
import LoadingSpinner from './common/LoadingSpinner';
import Card from './ui/Card';

export default function PdfUpload() {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const { uploadPdf, loading, metadata, error, setError } = useDocument();
  const { chunkSize, overlap } = useSettings();

  const handleFile = async (file) => {
    if (!file) return;
    setError(null);
    try {
      await uploadPdf(file, chunkSize, overlap);
    } catch {
      /* error set in context */
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files?.[0]);
  };

  return (
    <div className="mb-8">
      <Card
        className={`cursor-pointer border-2 border-dashed p-8 text-center transition-colors duration-200 ${
          dragOver
            ? 'border-[var(--accent)]'
            : 'border-[var(--border)] hover:border-[color-mix(in_srgb,var(--accent)_50%,var(--border))]'
        }`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          onChange={(e) => handleFile(e.target.files?.[0])}
          className="hidden"
        />
        <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-[var(--radius-xl)] bg-[var(--surface-hover)]">
          <svg
            className="h-7 w-7 text-[var(--accent)]"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
            />
          </svg>
        </div>
        <p className="text-sm font-semibold text-[var(--text)]">
          Drop your PDF here or{' '}
          <span className="text-[var(--accent)]">browse files</span>
        </p>
        <p className="mt-1.5 text-xs text-[var(--text-muted)]">
          We&apos;ll index it so you can summarize and ask questions
        </p>
        {loading && (
          <div className="mt-4 flex justify-center">
            <LoadingSpinner label="Getting your paper ready..." />
          </div>
        )}
      </Card>

      {metadata && !loading && (
        <p className="mt-4 text-sm text-[var(--accent)]">
          Ready: <strong className="text-[var(--text)]">{metadata.paper_name}</strong>
          {' · '}
          {metadata.page_count} pages
          {metadata.has_markdown ? ' · Markdown available' : ''}
        </p>
      )}
      {error && <p className="mt-2 text-sm text-[var(--danger)]">{error}</p>}
    </div>
  );
}
