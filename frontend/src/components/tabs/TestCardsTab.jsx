import { useEffect, useMemo, useState } from 'react';
import {
  generateTestCardAnswer,
  generateTestCards,
  verifyTestCards,
} from '../../api/client';
import { useDocument } from '../../context/DocumentContext';
import { useSettings } from '../../hooks/useSettings';
import ErrorBanner from '../common/ErrorBanner';
import LoadingSpinner from '../common/LoadingSpinner';
import Button from '../ui/Button';
import Card from '../ui/Card';
import Slider from '../ui/Slider';

function scoreBadgeClass(score) {
  if (score >= 80) return 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400';
  if (score >= 50) return 'bg-amber-500/15 text-amber-600 dark:text-amber-400';
  return 'bg-red-500/15 text-red-600 dark:text-red-400';
}

export default function TestCardsTab() {
  const doc = useDocument();
  const settings = useSettings();
  const [numCards, setNumCards] = useState(5);
  const [answers, setAnswers] = useState({});
  const [localError, setLocalError] = useState(null);
  const [generating, setGenerating] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [generatingAnswerId, setGeneratingAnswerId] = useState(null);

  useEffect(() => {
    setAnswers({});
  }, [doc.testCards]);

  const resultsByCardId = useMemo(() => {
    const map = {};
    (doc.testCardResults || []).forEach((r) => {
      map[r.card_id] = r;
    });
    return map;
  }, [doc.testCardResults]);

  const hasAnyAnswer = Object.values(answers).some((a) => a?.trim());

  const onGenerate = async () => {
    if (!doc.documentId) return;
    setGenerating(true);
    setLocalError(null);
    setAnswers({});
    doc.setTestCardResults(null);
    try {
      const data = await generateTestCards(doc.documentId, {
        num_cards: numCards,
        instruct_model_key: settings.instructModelKey,
      });
      doc.setTestCards(data.cards || []);
      const warnings = [data.generation_warning, data.instruct_warning].filter(Boolean);
      if (warnings.length) setLocalError(warnings.join(' '));
    } catch (err) {
      setLocalError(err.response?.data?.detail || err.message);
    } finally {
      setGenerating(false);
    }
  };

  const onGenerateAnswer = async (cardId) => {
    if (!doc.documentId) return;
    setGeneratingAnswerId(cardId);
    setLocalError(null);
    try {
      const data = await generateTestCardAnswer(doc.documentId, {
        card_id: cardId,
        instruct_model_key: settings.instructModelKey,
        qa_top_k: settings.qaTopK,
      });
      setAnswers((prev) => ({ ...prev, [cardId]: data.answer || '' }));
      if (data.instruct_warning) setLocalError(data.instruct_warning);
    } catch (err) {
      setLocalError(err.response?.data?.detail || err.message);
    } finally {
      setGeneratingAnswerId(null);
    }
  };

  const onVerify = async () => {
    if (!doc.documentId || !doc.testCards.length) return;
    setVerifying(true);
    setLocalError(null);
    try {
      const payload = {
        answers: doc.testCards.map((card) => ({
          card_id: card.id,
          answer: answers[card.id] || '',
        })),
        instruct_model_key: settings.instructModelKey,
      };
      const data = await verifyTestCards(doc.documentId, payload);
      doc.setTestCardResults(data.results || []);
      if (data.instruct_warning) setLocalError(data.instruct_warning);
    } catch (err) {
      setLocalError(err.response?.data?.detail || err.message);
    } finally {
      setVerifying(false);
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
    <div className="space-y-6">
      <div>
        <h3 className="section-title">Test your understanding</h3>
        <p className="mt-2 text-sm text-[var(--text-muted)]">
          Generate question cards from the paper, write your answers (or use Generate answer for
          Q&A help), then verify for an LLM-graded score.
        </p>
      </div>

      {localError && <ErrorBanner message={localError} />}

      <Card className="flex flex-wrap items-end gap-6">
        <Slider
          label="Number of cards"
          min={3}
          max={8}
          step={1}
          value={numCards}
          onChange={setNumCards}
          className="min-w-[12rem] flex-1"
        />
        <Button onClick={onGenerate} disabled={generating}>
          {generating ? 'Generating...' : 'Generate question cards'}
        </Button>
      </Card>

      {generating && <LoadingSpinner label="Generating test cards..." />}

      {doc.testCards.length > 0 && (
        <>
          <div className="grid gap-4 sm:grid-cols-2">
            {doc.testCards.map((card, index) => {
              const result = resultsByCardId[card.id];
              const isGeneratingAnswer = generatingAnswerId === card.id;
              return (
                <Card key={card.id} interactive>
                  <div className="mb-3 flex items-start justify-between gap-2">
                    <p className="text-sm font-semibold leading-snug">
                      {index + 1}. {card.question}
                    </p>
                    {result && (
                      <span
                        className={`shrink-0 rounded-lg px-2 py-0.5 text-xs font-bold ${scoreBadgeClass(result.score)}`}
                      >
                        {result.score}%
                      </span>
                    )}
                  </div>
                  <textarea
                    value={answers[card.id] || ''}
                    onChange={(e) =>
                      setAnswers((prev) => ({ ...prev, [card.id]: e.target.value }))
                    }
                    rows={4}
                    placeholder="Write your answer..."
                    className="input-field w-full resize-y"
                    disabled={isGeneratingAnswer}
                  />
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <Button
                      variant="ghost"
                      onClick={() => onGenerateAnswer(card.id)}
                      disabled={
                        generating ||
                        verifying ||
                        isGeneratingAnswer ||
                        generatingAnswerId !== null
                      }
                    >
                      {isGeneratingAnswer ? 'Generating answer...' : 'Generate answer'}
                    </Button>
                    {isGeneratingAnswer && (
                      <span className="text-xs text-[var(--text-muted)]">Using Q&A retrieval...</span>
                    )}
                  </div>
                  {result && (
                    <details className="mt-3 text-xs text-[var(--text-muted)]">
                      <summary className="cursor-pointer font-semibold text-[var(--text)]">
                        Grading details
                      </summary>
                      <div className="mt-2 space-y-2">
                        {result.judge_feedback && (
                          <p>
                            <span className="font-medium text-[var(--text)]">Feedback:</span>{' '}
                            {result.judge_feedback}
                          </p>
                        )}
                        {result.scoring_method === 'keyword_fallback' && (
                          <p className="text-[var(--warning)]">
                            Scored via keyword fallback (LLM judge unavailable).
                          </p>
                        )}
                        {result.matched_phrases?.length > 0 && (
                          <p>
                            <span className="font-medium text-emerald-600 dark:text-emerald-400">
                              Matched:
                            </span>{' '}
                            {result.matched_phrases.join(', ')}
                          </p>
                        )}
                        {result.missed_phrases?.length > 0 && (
                          <p>
                            <span className="font-medium text-red-600 dark:text-red-400">
                              Missed:
                            </span>{' '}
                            {result.missed_phrases.join(', ')}
                          </p>
                        )}
                        {result.reference_answer && (
                          <p>
                            <span className="font-medium text-[var(--text)]">Reference:</span>{' '}
                            {result.reference_answer}
                          </p>
                        )}
                      </div>
                    </details>
                  )}
                </Card>
              );
            })}
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <Button variant="ghost" onClick={onVerify} disabled={verifying || !hasAnyAnswer}>
              {verifying ? 'Grading answers...' : 'Verify answers'}
            </Button>
            {doc.testCardResults?.length > 0 && (
              <p className="text-sm font-semibold">
                Total score:{' '}
                <span className="text-[var(--accent)]">
                  {Math.round(
                    doc.testCardResults.reduce((sum, r) => sum + r.score, 0) /
                      doc.testCardResults.length,
                  )}
                  %
                </span>
              </p>
            )}
          </div>
          {verifying && <LoadingSpinner label="Grading answers with LLM judge..." />}
        </>
      )}

      {!generating && doc.testCards.length === 0 && (
        <p className="text-sm text-[var(--text-muted)]">
          Press &quot;Generate question cards&quot; to start.
        </p>
      )}
    </div>
  );
}
