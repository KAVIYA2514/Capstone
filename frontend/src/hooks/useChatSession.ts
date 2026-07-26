import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { chatApi, papersApi, type ChatRequest, type ChatResponse, type Source } from '../api/client'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  retrieval_strategy?: string
  embedding_model?: string
  isLoading?: boolean
}

export interface ChatSettings {
  retrieval_strategy: 'dense' | 'hybrid' | 'hybrid_rerank'
  embedding_model: string
}

export function useChatSession() {
  const [conversationId, setConversationId] = useState<string | undefined>()
  const [messages, setMessages] = useState<ChatMessage[]>([])

  const chatMutation = useMutation({
    mutationFn: (req: ChatRequest) => chatApi.send(req),
    onSuccess: (data: ChatResponse) => {
      setConversationId(data.conversation_id)
      // Replace the loading bubble with the actual response
      setMessages((prev) => {
        const updated = [...prev]
        const loadingIdx = updated.findLastIndex((m) => m.isLoading)
        if (loadingIdx !== -1) {
          updated[loadingIdx] = {
            id: `assistant-${Date.now()}`,
            role: 'assistant',
            content: data.answer,
            sources: data.sources,
            retrieval_strategy: data.retrieval_strategy,
            embedding_model: data.embedding_model,
          }
        }
        return updated
      })
    },
    onError: (err: Error) => {
      setMessages((prev) => {
        const updated = [...prev]
        const loadingIdx = updated.findLastIndex((m) => m.isLoading)
        if (loadingIdx !== -1) {
          updated[loadingIdx] = {
            id: `error-${Date.now()}`,
            role: 'assistant',
            content: `⚠️ Error: ${err.message}. Please try again.`,
          }
        }
        return updated
      })
    },
  })

  const sendMessage = (content: string, settings: ChatSettings) => {
    if (!content.trim()) return

    // Add user message immediately
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content,
    }
    // Add loading placeholder
    const loadingMsg: ChatMessage = {
      id: `loading-${Date.now()}`,
      role: 'assistant',
      content: '',
      isLoading: true,
    }
    setMessages((prev) => [...prev, userMsg, loadingMsg])

    chatMutation.mutate({
      message: content,
      conversation_id: conversationId,
      retrieval_strategy: settings.retrieval_strategy,
      embedding_model: settings.embedding_model,
    })
  }

  const resetSession = () => {
    setConversationId(undefined)
    setMessages([])
  }

  return {
    messages,
    conversationId,
    sendMessage,
    resetSession,
    isLoading: chatMutation.isPending,
  }
}

export function usePapers() {
  return useQuery({
    queryKey: ['papers'],
    queryFn: papersApi.list,
    refetchInterval: 10_000, // poll every 10s so newly ingested papers appear
  })
}

export function useUploadPaper() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ file, strategy }: { file: File; strategy?: string }) =>
      papersApi.upload(file, strategy),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['papers'] })
    },
  })
}

export function useDeletePaper() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => papersApi.delete(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['papers'] }),
  })
}
