import { useState, useCallback } from 'react'
import { Upload, X, FileText, Loader2, Check } from 'lucide-react'
import { useUploadPaper } from '../hooks/useChatSession'
import clsx from 'clsx'

type ChunkStrategy = 'fixed_512' | 'recursive' | 'semantic'

const STRATEGIES: { value: ChunkStrategy; label: string; desc: string }[] = [
  { value: 'fixed_512', label: 'Fixed-512', desc: '512-token windows' },
  { value: 'recursive', label: 'Recursive', desc: 'Paragraph-aware splits' },
  { value: 'semantic',  label: 'Semantic',  desc: 'Similarity-guided' },
]

export function PaperUploader() {
  const [isDragging, setIsDragging] = useState(false)
  const [strategy, setStrategy] = useState<ChunkStrategy>('recursive')
  const uploadMutation = useUploadPaper()

  const handleFiles = (files: FileList | null) => {
    if (!files) return
    Array.from(files).forEach((file) => {
      if (file.type === 'application/pdf') {
        uploadMutation.mutate({ file, strategy })
      }
    })
  }

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      handleFiles(e.dataTransfer.files)
    },
    [strategy]
  )

  const status = uploadMutation.isPending
    ? 'uploading'
    : uploadMutation.isSuccess
    ? 'success'
    : uploadMutation.isError
    ? 'error'
    : 'idle'

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-1">
        Upload Paper
      </h3>

      {/* Chunking strategy selector */}
      <div className="grid grid-cols-3 gap-1 p-1 glass rounded-xl">
        {STRATEGIES.map((s) => (
          <button
            key={s.value}
            onClick={() => setStrategy(s.value)}
            className={clsx(
              'text-[10px] font-medium px-2 py-1.5 rounded-lg transition-all duration-200',
              strategy === s.value
                ? 'bg-brand-500 text-white'
                : 'text-slate-500 hover:text-slate-300'
            )}
            title={s.desc}
          >
            {s.label}
          </button>
        ))}
      </div>

      {/* Drop zone */}
      <label
        className={clsx(
          'flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed',
          'px-4 py-6 cursor-pointer transition-all duration-200',
          isDragging
            ? 'border-brand-500/60 bg-brand-500/10'
            : 'border-white/10 hover:border-brand-500/40 hover:bg-white/5'
        )}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
      >
        <input
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
          disabled={uploadMutation.isPending}
        />

        {status === 'uploading' && (
          <>
            <Loader2 size={20} className="text-brand-400 animate-spin" />
            <span className="text-xs text-slate-500">Uploading & ingesting…</span>
          </>
        )}
        {status === 'success' && (
          <>
            <Check size={20} className="text-emerald-400" />
            <span className="text-xs text-emerald-400">Upload started!</span>
            <span className="text-[10px] text-slate-600">Chunks ready in ~60s</span>
          </>
        )}
        {status === 'error' && (
          <>
            <X size={20} className="text-red-400" />
            <span className="text-xs text-red-400">Upload failed</span>
          </>
        )}
        {status === 'idle' && (
          <>
            <Upload size={20} className="text-slate-600" />
            <span className="text-xs text-slate-500 text-center">
              Drop PDF here or <span className="text-brand-400">browse</span>
            </span>
            <span className="text-[10px] text-slate-700">Chunking: {strategy}</span>
          </>
        )}
      </label>
    </div>
  )
}
