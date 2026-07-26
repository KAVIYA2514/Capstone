import { BookOpen, Trash2, RefreshCw, Layers } from 'lucide-react'
import { usePapers, useDeletePaper } from '../hooks/useChatSession'
import type { Paper } from '../api/client'

function PaperRow({ paper }: { paper: Paper }) {
  const deleteMutation = useDeletePaper()

  return (
    <div className="flex items-start gap-3 p-3 glass rounded-xl hover:border-white/10 transition-all duration-200 group animate-fade-in">
      <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-brand-500/10 border border-brand-500/20 flex items-center justify-center">
        <BookOpen size={14} className="text-brand-400" />
      </div>

      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-slate-200 truncate leading-snug">
          {paper.title}
        </p>
        <div className="flex items-center gap-3 mt-1">
          {paper.total_pages && (
            <span className="text-[10px] text-slate-600">
              {paper.total_pages} pages
            </span>
          )}
          {paper.chunk_count > 0 && (
            <span className="flex items-center gap-1 text-[10px] text-slate-600">
              <Layers size={9} />
              {paper.chunk_count} chunks
            </span>
          )}
        </div>
      </div>

      <button
        onClick={() => deleteMutation.mutate(paper.id)}
        disabled={deleteMutation.isPending}
        className="opacity-0 group-hover:opacity-100 text-slate-700 hover:text-red-400
                   transition-all duration-200 p-1 rounded-lg hover:bg-red-400/10 flex-shrink-0"
        title="Delete paper"
      >
        <Trash2 size={13} />
      </button>
    </div>
  )
}

export function PaperLibrary() {
  const { data: papers, isLoading, isError, refetch } = usePapers()

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between px-1">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Paper Library
        </h3>
        <button
          onClick={() => refetch()}
          className="text-slate-700 hover:text-slate-400 transition-colors"
          title="Refresh"
        >
          <RefreshCw size={12} />
        </button>
      </div>

      {isLoading && (
        <div className="space-y-2">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-16 glass rounded-xl animate-pulse-slow" />
          ))}
        </div>
      )}

      {isError && (
        <p className="text-xs text-red-400 text-center py-4">
          Failed to load papers.
        </p>
      )}

      {papers && papers.length === 0 && (
        <p className="text-xs text-slate-600 text-center py-6 leading-relaxed">
          No papers ingested yet.<br />Upload a PDF above to get started.
        </p>
      )}

      {papers && papers.length > 0 && (
        <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
          {papers.map((paper) => (
            <PaperRow key={paper.id} paper={paper} />
          ))}
        </div>
      )}
    </div>
  )
}
