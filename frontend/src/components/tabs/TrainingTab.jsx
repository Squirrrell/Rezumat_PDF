import { useCallback, useEffect, useRef, useState } from 'react';
import {
  cancelTraining,
  evaluateTraining,
  fetchTrainingInfo,
  fetchTrainingJob,
  startTraining,
} from '../../api/client';
import ErrorBanner from '../common/ErrorBanner';
import LoadingSpinner from '../common/LoadingSpinner';
import Button from '../ui/Button';
import Card from '../ui/Card';
import Select from '../ui/Select';
import Slider from '../ui/Slider';

const ACTIVE_STATUSES = ['queued', 'running'];

const PRESETS = {
  smoke: { label: 'Smoke test (fast)', train_size: 50, val_size: 10, epochs: 2 },
  thesis: { label: 'Thesis experiment', train_size: 1000, val_size: 100, epochs: 5 },
};

function evalLossRows(logHistory) {
  return (logHistory || [])
    .filter((entry) => entry.eval_loss !== undefined)
    .map((entry) => ({
      epoch: Number(entry.epoch ?? 0),
      evalLoss: Number(entry.eval_loss),
    }));
}

function RougeTable({ models }) {
  if (!models) return null;
  const pretrained = models.pretrained;
  const finetuned = models.finetuned;
  const metrics = ['rouge1', 'rouge2', 'rougeL'];
  const labels = { rouge1: 'ROUGE-1', rouge2: 'ROUGE-2', rougeL: 'ROUGE-L' };

  return (
    <Card className="overflow-hidden p-0">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-[var(--border)] text-left">
            <th className="p-3 font-medium">Metric</th>
            {pretrained && <th className="p-3 font-medium">Pretrained</th>}
            {finetuned && <th className="p-3 font-medium">Fine-tuned</th>}
            {pretrained && finetuned && <th className="p-3 font-medium">Δ</th>}
          </tr>
        </thead>
        <tbody>
          {metrics.map((m) => {
            const delta =
              pretrained && finetuned ? finetuned[m] - pretrained[m] : null;
            return (
              <tr key={m} className="border-t border-[var(--border)] first:border-t-0">
                <td className="p-3 font-medium">{labels[m]}</td>
                {pretrained && (
                  <td className="p-3 font-mono text-[var(--text-muted)]">
                    {pretrained[m].toFixed(4)}
                  </td>
                )}
                {finetuned && (
                  <td className="p-3 font-mono text-[var(--text-muted)]">
                    {finetuned[m].toFixed(4)}
                  </td>
                )}
                {pretrained && finetuned && (
                  <td
                    className={`p-3 font-mono ${
                      delta >= 0
                        ? 'text-emerald-600 dark:text-emerald-400'
                        : 'text-red-600 dark:text-red-400'
                    }`}
                  >
                    {delta >= 0 ? '+' : ''}
                    {delta.toFixed(4)}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </Card>
  );
}

export default function TrainingTab() {
  const [info, setInfo] = useState(null);
  const [datasetConfig, setDatasetConfig] = useState('arxiv');
  const [trainSize, setTrainSize] = useState(1000);
  const [valSize, setValSize] = useState(100);
  const [epochs, setEpochs] = useState(5);

  const [job, setJob] = useState(null);
  const [starting, setStarting] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [rouge, setRouge] = useState(null);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  const isActive = job && ACTIVE_STATUSES.includes(job.status);

  const loadInfo = useCallback(async () => {
    try {
      const data = await fetchTrainingInfo();
      setInfo(data);
      if (data.last_results) setRouge(data.last_results);
      if (data.active_job) setJob(data.active_job);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  }, []);

  useEffect(() => {
    loadInfo();
  }, [loadInfo]);

  useEffect(() => {
    if (!job || !ACTIVE_STATUSES.includes(job.status)) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return undefined;
    }
    pollRef.current = setInterval(async () => {
      try {
        const data = await fetchTrainingJob(job.job_id);
        setJob(data);
      } catch {
        // keep last known state on transient errors
      }
    }, 2500);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
    };
  }, [job?.job_id, job?.status]);

  useEffect(() => {
    if (job && !ACTIVE_STATUSES.includes(job.status)) {
      loadInfo();
    }
  }, [job?.status, loadInfo]);

  const applyPreset = (key) => {
    const preset = PRESETS[key];
    setTrainSize(preset.train_size);
    setValSize(preset.val_size);
    setEpochs(preset.epochs);
  };

  const onStart = async () => {
    setStarting(true);
    setError(null);
    setRouge(null);
    try {
      const data = await startTraining({
        dataset_config: datasetConfig,
        train_size: trainSize,
        val_size: valSize,
        epochs,
      });
      setJob(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setStarting(false);
    }
  };

  const onCancel = async () => {
    if (!job) return;
    try {
      const data = await cancelTraining(job.job_id);
      setJob(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  const onEvaluate = async () => {
    setEvaluating(true);
    setError(null);
    try {
      const data = await evaluateTraining({
        dataset_config: datasetConfig,
        eval_size: valSize,
        run_baseline: true,
        run_finetuned: true,
      });
      setRouge(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setEvaluating(false);
    }
  };

  const depsMissing = info && !info.dependencies_installed;
  const lossRows = evalLossRows(job?.log_history);
  const progressPct = job?.total_epochs
    ? Math.min(100, Math.round((job.current_epoch / job.total_epochs) * 100))
    : 0;

  return (
    <div className="space-y-6">
      <div>
        <h3 className="section-title">Fine-tune T5-small (offline experiment)</h3>
        <p className="mt-2 max-w-2xl text-sm text-[var(--text-muted)]">
          Fine-tune the pretrained T5-small summarizer on a streamed subset of the
          scientific_papers dataset (article {'->'} abstract), then compare ROUGE against the
          pretrained baseline. This is a research experiment and does not change the
          summarizer used elsewhere in the app.
        </p>
      </div>

      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

      {info && (
        <Card className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
          <span>
            <span className="text-[var(--text-muted)]">Device: </span>
            <span className="font-medium">
              {info.device}
              {info.gpu_name ? ` (${info.gpu_name})` : ''}
            </span>
          </span>
          <span>
            <span className="text-[var(--text-muted)]">CUDA: </span>
            <span className="font-medium">{info.cuda_available ? 'available' : 'CPU only'}</span>
          </span>
          <span>
            <span className="text-[var(--text-muted)]">Default dataset: </span>
            <span className="font-medium">scientific_papers / {info.default_dataset_config}</span>
          </span>
        </Card>
      )}

      {depsMissing && <ErrorBanner message={info.dependencies_message} />}

      <Card className="space-y-5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-[var(--text-muted)]">Presets:</span>
          {Object.entries(PRESETS).map(([key, preset]) => (
            <Button key={key} variant="ghost" onClick={() => applyPreset(key)} disabled={isActive}>
              {preset.label}
            </Button>
          ))}
        </div>

        <div className="grid gap-6 sm:grid-cols-2">
          <Select
            label="Dataset"
            value={datasetConfig}
            onChange={(e) => setDatasetConfig(e.target.value)}
            disabled={isActive}
          >
            <option value="arxiv">scientific_papers / arxiv</option>
            <option value="pubmed">scientific_papers / pubmed</option>
          </Select>

          <Slider
            label="Epochs"
            min={1}
            max={20}
            step={1}
            value={epochs}
            onChange={setEpochs}
          />

          <label className="block text-xs text-[var(--text-muted)]">
            <span className="mb-1.5 block font-medium text-[var(--text)]">Train examples</span>
            <input
              type="number"
              min={10}
              max={20000}
              value={trainSize}
              onChange={(e) => setTrainSize(Number(e.target.value))}
              disabled={isActive}
              className="input-field w-full"
            />
          </label>

          <label className="block text-xs text-[var(--text-muted)]">
            <span className="mb-1.5 block font-medium text-[var(--text)]">Validation examples</span>
            <input
              type="number"
              min={5}
              max={2000}
              value={valSize}
              onChange={(e) => setValSize(Number(e.target.value))}
              disabled={isActive}
              className="input-field w-full"
            />
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={onStart} disabled={starting || isActive || depsMissing}>
            {starting ? 'Starting...' : 'Start training'}
          </Button>
          {isActive && (
            <Button variant="ghost" onClick={onCancel}>
              Cancel
            </Button>
          )}
          <p className="text-xs text-[var(--text-muted)]">
            Training can take a long time on CPU. Jobs run in-memory and are lost on backend
            restart.
          </p>
        </div>
      </Card>

      {job && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="section-title">Progress</h3>
            <span className="rounded-lg bg-[var(--surface-hover)] px-2 py-0.5 text-xs font-semibold uppercase tracking-wide">
              {job.status}
            </span>
          </div>

          {isActive && (
            <>
              <LoadingSpinner
                label={`Epoch ${job.current_epoch.toFixed(2)} / ${job.total_epochs} (step ${job.current_step})`}
              />
              <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--surface-hover)]">
                <div
                  className="h-full rounded-full bg-[var(--accent)] transition-all"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </>
          )}

          {job.status === 'failed' && job.error && <ErrorBanner message={job.error} />}

          {lossRows.length > 0 && (
            <Card className="overflow-hidden p-0">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="border-b border-[var(--border)] text-left">
                    <th className="p-3 font-medium">Epoch</th>
                    <th className="p-3 font-medium">Eval loss</th>
                  </tr>
                </thead>
                <tbody>
                  {lossRows.map((row) => (
                    <tr
                      key={row.epoch}
                      className="border-t border-[var(--border)] first:border-t-0"
                    >
                      <td className="p-3">{row.epoch.toFixed(2)}</td>
                      <td className="p-3 font-mono text-[var(--text-muted)]">
                        {row.evalLoss.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {job.status === 'completed' && job.checkpoint_path && (
            <p className="text-sm text-[var(--text-muted)]">
              Saved fine-tuned model to{' '}
              <span className="font-mono text-[var(--text)]">{job.checkpoint_path}</span>
            </p>
          )}
        </section>
      )}

      <section className="space-y-4">
        <h3 className="section-title">ROUGE evaluation</h3>
        <p className="text-sm text-[var(--text-muted)]">
          Compare the pretrained baseline against your latest fine-tuned checkpoint on the
          validation slice.
        </p>
        <Button
          variant="ghost"
          onClick={onEvaluate}
          disabled={evaluating || isActive || depsMissing || !info?.final_checkpoint}
        >
          {evaluating ? 'Evaluating...' : 'Run ROUGE evaluation'}
        </Button>
        {!info?.final_checkpoint && (
          <p className="text-xs text-[var(--text-muted)]">
            Train a model first to enable fine-tuned evaluation.
          </p>
        )}
        {evaluating && <LoadingSpinner label="Generating summaries and scoring ROUGE..." />}
        {rouge?.models && (
          <>
            <p className="text-xs text-[var(--text-muted)]">
              Dataset: scientific_papers / {rouge.dataset_config} | N = {rouge.eval_size}
            </p>
            <RougeTable models={rouge.models} />
          </>
        )}
      </section>
    </div>
  );
}
