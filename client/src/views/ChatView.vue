<template>
  <div class="main-content">
    <div class="chat-section" :class="{ 'has-evidence': panelEvidences.length > 0 && !evidenceCollapsed }">

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
          <MessageBubble
            :role="msg.role"
            :content="msg.content"
            :intent="msg.intent"
            :threadId="store.activeThreadId"
          />
        </template>

        <MessageBubble
          v-if="store.isStreaming"
          role="assistant"
          :content="store.streamingText"
          streaming
          :intent="store.currentIntent"
          :threadId="store.activeThreadId"
        />

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
    <div class="evidence-section" v-if="panelEvidences.length > 0" :class="{ collapsed: evidenceCollapsed }">
      <!-- Toggle bar (always visible when panel exists) -->
      <div class="evidence-toggle" @click="evidenceCollapsed = !evidenceCollapsed">
        <span class="toggle-icon">{{ evidenceCollapsed ? '▶' : '▼' }}</span>
        <span class="toggle-text">引用证据</span>
        <span class="toggle-count">{{ panelEvidences.length }} 条</span>
      </div>
      <!-- Collapsible content -->
      <EvidencePanel v-show="!evidenceCollapsed" :evidences="panelEvidences" />
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
    const evidenceCollapsed = ref(false)

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

    return { store, input, send, msgList, inputEl, panelEvidences, evidenceCollapsed }
  },
}
</script>

<style scoped>
.evidence-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  cursor: pointer;
  user-select: none;
  transition: background 0.15s;
  border-bottom: 1px solid var(--color-hairline, #eee);
}

.evidence-toggle:hover {
  background: rgba(86, 69, 212, 0.05);
}

.toggle-icon {
  font-size: 10px;
  color: var(--color-text-secondary, #888);
  width: 14px;
  text-align: center;
  flex-shrink: 0;
}

.toggle-text {
  font-weight: 600;
  font-size: var(--fs-body, 14px);
  color: var(--color-text, #333);
}

.toggle-count {
  font-size: 12px;
  color: var(--color-text-secondary, #888);
  background: var(--color-surface, #f0f0f0);
  padding: 0 8px;
  border-radius: 10px;
  line-height: 20px;
  margin-left: auto;
}

/* 折叠时面板缩小到仅显示 toggle 栏 */
.evidence-section.collapsed {
  flex: 0 0 auto;
  overflow: hidden;
}

.evidence-section.collapsed .evidence-toggle {
  border-bottom: none;
}
</style>
