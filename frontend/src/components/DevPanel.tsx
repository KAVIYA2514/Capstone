import { Settings2 } from 'lucide-react'
import clsx from 'clsx'
import type { ChatSettings } from '../hooks/useChatSession'

interface DevPanelProps {
  settings: ChatSettings
  onChange: (s: ChatSettings) => void
}

const STRATEGIES: { value: ChatSettings['retrieval_strategy']; label: string; desc: string }[] = [
  { value: 'dense',         label: 'Dense',         desc: 'pgvector cosine similarity' },
  { value: 'hybrid',        label: 'Hybrid RRF',    desc: 'Dense + full-text fusion' },
  { value: 'hybrid_rerank', label: 'Hybrid+Rerank', desc: 'Hybrid + cross-encoder' },
]

const EMBEDDING_MODELS: { value: string; label: string; desc: string }[] = [
  { value: 'nvidia/nv-embedqa-e5-v5', label: 'NVIDIA E5-v5', desc: '1024-dim · API' },
  { value: 'BAAI/bge-m3',             label: 'BGE-M3',       desc: '1024-dim · Local' },
]

export function DevPanel({ settings, onChange }: DevPanelProps) {
  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2 px-1">
        <Settings2 size={13} className="text-brand-400" />
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Dev Panel
        </h3>
      </div>

      {/* Retrieval strategy */}
      <div className="space-y-2">
        <p className="text-[10px] text-slate-600 uppercase tracking-wider px-1">
          Retrieval Strategy
        </p>
        <div className="space-y-1">
          {STRATEGIES.map((s) => (
            <button
              key={s.value}
              onClick={() => onChange({ ...settings, retrieval_strategy: s.value })}
              className={clsx(
                'w-full text-left px-3 py-2.5 rounded-xl transition-all duration-200 border',
                settings.retrieval_strategy === s.value
                  ? 'bg-brand-500/15 border-brand-500/40 text-brand-300'
                  : 'border-transparent text-slate-500 hover:text-slate-300 hover:bg-white/5'
              )}
              id={`strategy-${s.value}`}
            >
              <div className="text-xs font-medium">{s.label}</div>
              <div className="text-[10px] opacity-60 mt-0.5">{s.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Embedding model */}
      <div className="space-y-2">
        <p className="text-[10px] text-slate-600 uppercase tracking-wider px-1">
          Embedding Model
        </p>
        <div className="space-y-1">
          {EMBEDDING_MODELS.map((m) => (
            <button
              key={m.value}
              onClick={() => onChange({ ...settings, embedding_model: m.value })}
              className={clsx(
                'w-full text-left px-3 py-2.5 rounded-xl transition-all duration-200 border',
                settings.embedding_model === m.value
                  ? 'bg-violet-500/15 border-violet-500/40 text-violet-300'
                  : 'border-transparent text-slate-500 hover:text-slate-300 hover:bg-white/5'
              )}
              id={`model-${m.value.replace('/', '-')}`}
            >
              <div className="text-xs font-medium">{m.label}</div>
              <div className="text-[10px] opacity-60 mt-0.5">{m.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Active config badge */}
      <div className="glass rounded-xl p-3">
        <p className="text-[10px] text-slate-600 uppercase tracking-wider mb-2">Active Config</p>
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-600">Strategy</span>
            <span className="text-[10px] font-mono text-brand-400">
              {settings.retrieval_strategy}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-600">Embedder</span>
            <span className="text-[10px] font-mono text-violet-400 truncate max-w-[100px]">
              {settings.embedding_model.split('/')[1] || settings.embedding_model}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
