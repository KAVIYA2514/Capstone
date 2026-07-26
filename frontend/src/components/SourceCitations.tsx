import { useState } from 'react'
import { ChevronDown, ChevronUp, FileText, Hash } from 'lucide-react'
import type { Source } from '../api/client'
import clsx from 'clsx'

interface SourceCitationsProps {
  sources: Source[]
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const [expanded, setExpanded] = useState(false)

  const scoreColor =
    source.score > 0.7
      ? 'text-emerald-400'
      : source.score > 0.4
      ? 'text-amber-400'
      : 'text-slate-500'

  return (
    <div
      className="source-card animate-fade-in"
      style={{ animationDelay: `${index * 60}ms` }}
      onClick={() => setExpanded((e) => !e)}
    >
      <div className="flex items-start gap-3">
        {/* Index badge */}
        <div className="flex-shrink-0 w-6 h-6 rounded-full bg-brand-500/20 border border-brand-500/30 flex items-center justify-center">
          <span className="text-brand-300 text-xs font-bold">{index + 1}</span>
        </div>

        <div className="flex-1 min-w-0">
          {/* Paper title */}
          <div className="flex items-center gap-1.5 mb-0.5">
            <FileText size={12} className="text-brand-400 flex-shrink-0" />
            <span className="text-sm font-medium text-slate-200 truncate">
              {source.paper_title}
            </span>
          </div>

          {/* Page number + score */}
          <div className="flex items-center gap-3 text-xs">
            {source.page_number && (
              <span className="flex items-center gap-1 text-slate-500">
                <Hash size={10} />
                Page {source.page_number}
              </span>
            )}
            <span className={clsx('font-mono font-medium', scoreColor)}>
              {(source.score * 100).toFixed(0)}% relevant
            </span>
          </div>
        </div>

        {/* Expand toggle */}
        <div className="flex-shrink-0 text-slate-600 hover:text-slate-400 transition-colors">
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </div>
      </div>

      {/* Expanded chunk preview */}
      {expanded && (
        <div className="mt-3 pt-3 border-t border-white/5 animate-fade-in">
          <p className="text-xs text-slate-400 leading-relaxed font-mono">
            {source.chunk_text}
          </p>
        </div>
      )}
    </div>
  )
}

export function SourceCitations({ sources }: SourceCitationsProps) {
  if (!sources || sources.length === 0) return null

  return (
    <div className="mt-3 space-y-2">
      <p className="text-xs font-medium text-slate-500 uppercase tracking-wider px-1">
        Sources ({sources.length})
      </p>
      {sources.map((source, i) => (
        <SourceCard key={source.chunk_id ?? i} source={source} index={i} />
      ))}
    </div>
  )
}
