import { useState } from 'react'
import { ChatWindow } from '../components/ChatWindow'
import { DevPanel } from '../components/DevPanel'
import { PaperUploader } from '../components/PaperUploader'
import { PaperLibrary } from '../components/PaperLibrary'
import { useChatSession, type ChatSettings } from '../hooks/useChatSession'
import { ChevronLeft, ChevronRight } from 'lucide-react'

export function ChatPage() {
  const { messages, conversationId, sendMessage, resetSession, isLoading } = useChatSession()

  const [settings, setSettings] = useState<ChatSettings>({
    retrieval_strategy: 'hybrid',
    embedding_model: 'nvidia/nv-embedqa-e5-v5',
  })

  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className="h-screen flex bg-surface-950 overflow-hidden">
      {/* ── Left sidebar ── */}
      <aside
        className={`flex-shrink-0 transition-all duration-300 ease-in-out border-r border-white/5
                   ${sidebarOpen ? 'w-72' : 'w-0 overflow-hidden'}`}
      >
        <div className="w-72 h-full flex flex-col p-4 gap-6 overflow-y-auto">
          {/* App branding */}
          <div className="pt-2">
            <h1 className="text-lg font-bold gradient-text">RAG Research Bot</h1>
            <p className="text-[10px] text-slate-600 mt-0.5">
              Powered by NVIDIA NIM · pgvector
            </p>
          </div>

          <DevPanel settings={settings} onChange={setSettings} />
          <div className="border-t border-white/5 pt-6">
            <PaperUploader />
          </div>
          <div className="border-t border-white/5 pt-6 flex-1">
            <PaperLibrary />
          </div>
        </div>
      </aside>

      {/* ── Toggle sidebar button ── */}
      <button
        onClick={() => setSidebarOpen((o) => !o)}
        className="absolute left-0 top-1/2 -translate-y-1/2 z-10 w-5 h-10 bg-surface-800 border border-white/5 rounded-r-lg
                   flex items-center justify-center text-slate-600 hover:text-slate-300 transition-all"
        style={{ left: sidebarOpen ? '288px' : '0' }}
      >
        {sidebarOpen ? <ChevronLeft size={12} /> : <ChevronRight size={12} />}
      </button>

      {/* ── Main chat area ── */}
      <main className="flex-1 min-w-0 relative">
        {/* Ambient gradient background */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <div className="absolute -top-40 -right-40 w-96 h-96 bg-brand-500/5 rounded-full blur-3xl" />
          <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-violet-500/5 rounded-full blur-3xl" />
        </div>

        <div className="relative h-full">
          <ChatWindow
            messages={messages}
            isLoading={isLoading}
            settings={settings}
            conversationId={conversationId}
            onSend={sendMessage}
            onReset={resetSession}
          />
        </div>
      </main>
    </div>
  )
}
