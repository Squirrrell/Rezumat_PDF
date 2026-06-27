import { createContext, useContext, useMemo, useState } from 'react';

const SettingsContext = createContext(null);

export function SettingsProvider({ children }) {
  const [instructModelChoice, setInstructModelChoice] = useState('Qwen2.5-0.5B-Instruct');
  const [chunkSize, setChunkSize] = useState(1200);
  const [overlap, setOverlap] = useState(150);
  const [answerLength, setAnswerLength] = useState('medium');
  const [qaTopK, setQaTopK] = useState(5);
  const [summaryLength, setSummaryLength] = useState('medium');
  const [summaryMaxChunks, setSummaryMaxChunks] = useState(12);

  const instructModelKey = instructModelChoice.toLowerCase().includes('flan')
    ? 'flan'
    : 'qwen';

  const qaPayload = (question) => ({
    question,
    answer_length: answerLength,
    qa_top_k: qaTopK,
    use_instruct_qa: true,
    instruct_model_key: instructModelKey,
  });

  const testCardsPayload = {
    instruct_model_key: instructModelKey,
  };

  const summaryPayload = (source) => ({
    source,
    length: summaryLength,
    max_chunks: summaryMaxChunks,
  });

  const comprehensivePayload = (source) => ({
    source,
    instruct_model_key: instructModelKey,
    qa_top_k: qaTopK,
  });

  const value = {
    instructModelChoice,
    setInstructModelChoice,
    chunkSize,
    setChunkSize,
    overlap,
    setOverlap,
    answerLength,
    setAnswerLength,
    qaTopK,
    setQaTopK,
    summaryLength,
    setSummaryLength,
    summaryMaxChunks,
    setSummaryMaxChunks,
    instructModelKey,
    qaPayload,
    testCardsPayload,
    summaryPayload,
    comprehensivePayload,
  };

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('useSettings must be used within SettingsProvider');
  return ctx;
}
