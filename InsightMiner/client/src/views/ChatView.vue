<template>
  <div class="main-content">
    <div class="chat-section" :class="{ 'has-evidence': panelEvidences.length > 0 }">

      <!-- Chat header -->
      <div class="chat-header">
        <span class="thread-label">
          {{ store.activeThreadId ? '对话 ' + store.activeThreadId.slice(0, 8) + '...' : '新对话' }}
        </span>
        <span class="kb-badge">◈ {{ store.activeKbId }}</span>
      </div>

      <!-- Messages -->
      <div class="chat-messages" ref="msgList">

        <template v-for="(msg, i) in store.messages" :key="i">
          <MessageBubble :role="msg.role" :content="msg.content" :intent="msg.intent" />
        </template>

        <MessageBubble
          v-if="store.isStreaming"
          role="assistant"
          :content="store.streamingText"
          streaming
          :intent="store.currentIntent" />

        <div v-if="store.error" class="empty-state">
          <p style="color:var(--color-error);font-size:var(--fs-body-sm);">{{ store.error }}</p>
        </div>

        <div v-if="!store.messages.length && !store.isStreaming && !store.error" class="empty-state">
          <div class="icon">◈</div>
          <h2>InsightMiner</h2>
          <p>向知识库提问，获取基于文档的智能回答</p>
        </div>

      </div>

      <!-- Input area -->
      <div class="chat-input-area">
        <div class="chat-input-row">
          <textarea class="chat-input" v-model="input"
            placeholder="输入问题…"
            rows="1"
            @keydown.enter.exact.prevent="send"
            :disabled="store.isStreaming"
            ref="inputEl"></textarea>
          <button class="send-btn" @click="send" :disabled="store.isStreaming || !input.trim()">
            {{ store.isStreaming ? '...' : '发送' }}
          </button>
        </div>
      </div>

    </div>

    <!-- Evidence panel -->
    <div class="evidence-section" v-if="panelEvidences.length > 0">
      <EvidencePanel :evidences="panelEvidences" />
    </div>
  </div>
</template>

<script>
import MessageBubble from '@/components/MessageBubble.vue'
import EvidencePanel from '@/components/EvidencePanel.vue'
import { useChatStore } from '@/stores/chat'
import { computed, watch, ref, nextTick } from 'vue'

export default {
  components: { MessageBubble, EvidencePanel },
  setup() {
    const store = useChatStore()
    const input = ref('')
    const msgList = ref(null)
    const inputEl = ref(null)

    const panelEvidences = computed(() => {
      if (store.currentEvidences.length) return store.currentEvidences
      const msgs = store.messages
      if (msgs.length) {
        const last = msgs[msgs.length - 1]
        if (last.evidences) return last.evidences
      }
      return []
    })

    async function send() {
      const q = input.value.trim()
      if (!q || store.isStreaming) return
      input.value = ''
      await store.sendMessage(q)
      inputEl.value?.focus()
    }

    watch(
      () => store.messages.length + (store.isStreaming ? 1 : 0),
      async () => {
        await nextTick()
        const el = msgList.value
        if (el) el.scrollTop = el.scrollHeight
      },
      { flush: 'post' }
    )

    return { store, input, send, msgList, inputEl, panelEvidences }
  },
}
</script>
