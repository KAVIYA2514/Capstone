import { useEffect, useRef, useState } from 'react'
import { Send, RotateCcw, BookOpen } from 'lucide-react'
import { MessageBubble } from './MessageBubble'
import type { ChatMessage, ChatSettings } from '../hooks/useChatSession'

interface ChatWindowProps {
  messages: ChatMessage[]
  isLoading: boolean
  settings: ChatSettings
  conversationId: string | undefined
  onSend: (content: string, settings: ChatSettings) => void
  onReset: () => void
}

const WELCOME_QUERIES = [
  'What is the attention mechanism in Transformers?',
  'How does BERT use masked language modeling?',
  'Explain retrieval-augmented generation.',
  'How does LoRA reduce trainable parameters?',
  'What is chain-of-thought prompting?',
]

export function ChatWindow({
  messages,
  isLoading,
  settings,
  conversationId,
  onSend,
  onReset,
}: ChatWindowProps) {
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed || isLoading) return
    onSend(trimmed, settings)
    setInput('')
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-brand-500 to-violet-600 flex items-center justify-center">
            <BookOpen size={16} className="text-white" />
          </div>
          <div>
            <h1 className="text-sm font-semibold gradient-text">Research Paper Answer Bot</h1>
            {conversationId && (
              <p className="text-[10px] text-slate-600 font-mono truncate max-w-[180px]">
                {conversationId.slice(0, 8)}…
              </p>
            )}
          </div>
        </div>
        <button
          onClick={onReset}
          className="btn-ghost flex items-center gap-1.5 text-xs"
          title="Start new conversation"
        >
          <RotateCcw size={13} />
          New
        </button>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
        {isEmpty ? (
          // Welcome screen
          <div className="flex flex-col items-center justify-center h-full gap-8 animate-fade-in">
            <div className="text-center space-y-3">
              <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-brand-500/20 to-violet-600/20 border border-brand-500/20 flex items-center justify-center">
                <BookOpen size={32} className="text-brand-400" />
              </div>
              <h2 className="text-xl font-semibold gradient-text">Ask about Research Papers</h2>
              <p className="text-slate-500 text-sm max-w-xs leading-relaxed">
                Get grounded answers from GenAI research papers with source citations.
              </p>
            </div>

            {/* Suggested queries */}
            <div className="grid gap-2 w-full max-w-md">
              {WELCOME_QUERIES.map((q) => (
                <button
                  key={q}
                  onClick={() => onSend(q, settings)}
                  className="text-left text-sm text-slate-400 hover:text-slate-200 glass rounded-xl px-4 py-3
                             hover:border-brand-500/30 transition-all duration-200 hover:translate-x-1"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="px-6 py-4 border-t border-white/5">
        <div className="flex gap-3 items-end">
          <div className="flex-1 glass rounded-2xl overflow-hidden input-glow border border-white/5 transition-all">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about the research papers… (Enter to send)"
              rows={1}
              disabled={isLoading}
              className="w-full bg-transparent px-4 py-3.5 text-sm text-slate-100 placeholder-slate-600
                         resize-none outline-none leading-relaxed
                         disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ maxHeight: '120px' }}
              onInput={(e) => {
                const el = e.currentTarget
                el.style.height = 'auto'
                el.style.height = `${Math.min(el.scrollHeight, 120)}px`
              }}
            />
          </div>
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="btn-primary flex items-center justify-center w-11 h-11 rounded-xl flex-shrink-0"
            id="send-button"
          >
            <Send size={16} />
          </button>
        </div>
        <p className="text-[10px] text-slate-700 mt-2 text-center">
          Shift+Enter for new line · Answers grounded in paper context only
        </p>
      </div>
    </div>
  )
}
