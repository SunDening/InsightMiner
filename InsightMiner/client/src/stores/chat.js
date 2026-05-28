import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { createKB, listKBs, listDocs, uploadDoc, deleteDoc, listThreads, getThreadMessages, deleteThread, streamChat } from '@/services/api'

export const useChatStore = defineStore('chat', () => {
  // ── KB state ──
  const kbList = ref([])
  const activeKbId = ref('default')
  const docs = ref([])

  // ── Thread state ──
  const threads = ref([])
  const activeThreadId = ref(null)
  const messages = ref([])

  // ── Streaming state ──
  const isStreaming = ref(false)
  const streamingText = ref('')
  const currentEvidences = ref([])
  const currentIntent = ref('')
  const error = ref(null)

  const activeThreadMessages = computed(() => messages.value)

  // ── KB actions ──
  async function fetchKBs() {
    kbList.value = await listKBs()
    if (!kbList.value.find(k => k.kb_id === activeKbId.value)) {
      activeKbId.value = 'default'
      try { await createKB('default') } catch (_) { /* already exists */ }
    }
  }

  async function fetchDocs() {
    docs.value = await listDocs(activeKbId.value)
  }

  async function uploadFile(file) {
    const result = await uploadDoc(activeKbId.value, file)
    await fetchDocs()
    return result
  }

  async function removeDoc(filename) {
    await deleteDoc(activeKbId.value, filename)
    await fetchDocs()
  }

  async function switchKb(kbId) {
    activeKbId.value = kbId
    activeThreadId.value = null
    messages.value = []
    currentEvidences.value = []
    streamingText.value = ''
    error.value = null
    await fetchDocs()
    await fetchThreads()
  }

  // ── Thread actions ──
  async function fetchThreads() {
    threads.value = await listThreads(activeKbId.value)
  }

  async function switchThread(threadId) {
    if (isStreaming.value) return
    activeThreadId.value = threadId
    messages.value = []
    currentEvidences.value = []
    if (threadId) {
      messages.value = await getThreadMessages(threadId)
    }
  }

  function newThread() {
    if (isStreaming.value) return
    activeThreadId.value = null
    messages.value = []
    currentEvidences.value = []
    streamingText.value = ''
    error.value = null
  }

  async function removeThread(threadId) {
    await deleteThread(threadId)
    if (activeThreadId.value === threadId) newThread()
    await fetchThreads()
  }

  // ── Chat actions ──
  async function sendMessage(question) {
    if (!question.trim() || isStreaming.value) return

    error.value = null
    messages.value.push({ role: 'user', content: question })
    streamingText.value = ''
    currentEvidences.value = []
    currentIntent.value = ''
    isStreaming.value = true

    try {
      let currentThread = activeThreadId.value
      let fullAnswer = ''

      for await (const ev of streamChat(question, currentThread, activeKbId.value)) {
        if (ev.event === 'thread_id') {
          currentThread = ev.data.thread_id
          if (!activeThreadId.value) {
            activeThreadId.value = currentThread
            await fetchThreads()
          }
        } else if (ev.event === 'intent') {
          currentIntent.value = ev.data.intent || ''
        } else if (ev.event === 'evidence') {
          currentEvidences.value = ev.data.evidences || []
        } else if (ev.event === 'token') {
          fullAnswer += ev.data.token
          streamingText.value = fullAnswer
        } else if (ev.event === 'error') {
          error.value = ev.data.message
          isStreaming.value = false
          return
        } else if (ev.event === 'done') {
          // done
        }
      }

      if (fullAnswer) {
        messages.value.push({ role: 'assistant', content: fullAnswer, evidences: [...currentEvidences.value], intent: currentIntent.value })
      }
    } catch (e) {
      error.value = e.message || '请求失败'
    } finally {
      isStreaming.value = false
      streamingText.value = ''
      currentEvidences.value = []
      await fetchThreads()
    }
  }

  return {
    kbList, activeKbId, docs,
    threads, activeThreadId, messages,
    isStreaming, streamingText, currentEvidences, currentIntent, error,
    activeThreadMessages,
    fetchKBs, fetchDocs, uploadFile, removeDoc, switchKb,
    fetchThreads, switchThread, newThread, removeThread,
    sendMessage,
  }
})
