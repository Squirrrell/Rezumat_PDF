import { useState } from 'react';
import AppFooter from './components/AppFooter';
import AppHeader from './components/AppHeader';
import PdfUpload from './components/PdfUpload';
import SettingsDrawer from './components/SettingsDrawer';
import MarkdownTab from './components/tabs/MarkdownTab';
import MetricsTab from './components/tabs/MetricsTab';
import SummaryTab from './components/tabs/SummaryTab';
import TestCardsTab from './components/tabs/TestCardsTab';
import TrainingTab from './components/tabs/TrainingTab';
import { APP_NAME, APP_TAGLINE } from './constants';
import { useDocument } from './context/DocumentContext';

const TAB_COMPONENTS = {
  summary: SummaryTab,
  markdown: MarkdownTab,
  'test-cards': TestCardsTab,
  metrics: MetricsTab,
  training: TrainingTab,
};

export default function App() {
  const [activeTab, setActiveTab] = useState('summary');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [scrollToRuntime, setScrollToRuntime] = useState(false);
  const doc = useDocument();
  const ActiveComponent = TAB_COMPONENTS[activeTab];

  const totalTime = Object.values(doc.lastRuntime || {}).reduce(
    (a, b) => a + Number(b),
    0,
  );

  const openSettings = () => {
    setScrollToRuntime(false);
    setSettingsOpen(true);
  };

  const openRuntime = () => {
    setScrollToRuntime(true);
    setSettingsOpen(true);
  };

  const closeSettings = () => {
    setSettingsOpen(false);
    setScrollToRuntime(false);
  };

  return (
    <div className="flex min-h-screen flex-col">
      <AppHeader
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onOpenSettings={openSettings}
        onOpenRuntime={openRuntime}
      />

      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8 lg:px-10">
        <div className="mb-8">
          <h1 className="page-title">{APP_NAME}</h1>
          <p className="mt-2 max-w-2xl text-sm text-[var(--text-muted)]">
            {APP_TAGLINE}. Upload a paper, get a summary, and ask questions. Everything stays on
            your computer.
          </p>
        </div>

        <PdfUpload />
        <ActiveComponent />
      </main>

      <AppFooter metadata={doc.metadata} totalTime={totalTime} />

      <SettingsDrawer
        open={settingsOpen}
        onClose={closeSettings}
        scrollToRuntime={scrollToRuntime}
      />
    </div>
  );
}
