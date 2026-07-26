import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Bot, User } from 'lucide-react'
import { SourceCitations } from './SourceCitations'
import type { ChatMessage } from '../hooks/useChatSession'

interface MessageBubbleProps {
  message: ChatMessage
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-2 py-1">
      <div className="typing-dot" />
      <div className="typing-dot" />
      <div className="typing-dot" />
    </div>
  )
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <div
      className={`flex gap-3 animate-slide-up ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
    >
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center
        ${isUser
          ? 'bg-brand-500/20 border border-brand-500/30'
          : 'bg-violet-500/20 border border-violet-500/30'
        }`}
      >
        {isUser
          ? <User size={14} className="text-brand-300" />
          : <Bot size={14} className="text-violet-300" />
        }
      </div>

      {/* Content */}
      <div className={`flex-1 ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        {isUser ? (
          <div className="bubble-user">
            <p className="text-sm leading-relaxed">{message.content}</p>
          </div>
        ) : (
          <div className="bubble-assistant">
            {message.isLoading ? (
              <TypingIndicator />
            ) : (
              <>
                <div className="prose-chat">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                  </ReactMarkdown>
                </div>
                {message.sources && message.sources.length > 0 && (
                  <SourceCitations sources={message.sources} />
                )}
                {/* Strategy badge */}
                {message.retrieval_strategy && (
                  <div className="mt-3 pt-2 border-t border-white/5 flex items-center gap-2">
                    <span className="text-[10px] text-slate-600 uppercase tracking-wider">
                      {message.retrieval_strategy.replace('_', ' ')}
                    </span>
                    <span className="text-[10px] text-slate-700">·</span>
                    <span className="text-[10px] text-slate-600 font-mono truncate max-w-[160px]">
                      {message.embedding_model}
                    </span>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
