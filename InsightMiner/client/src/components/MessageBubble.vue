<template>
  <div class="message" :class="role">
    <div class="message-meta" v-if="role === 'assistant'">
      {{ streaming ? '思考中...' : 'AI' }}
    </div>
    <div class="message-content" :class="{ 'streaming-cursor': streaming }">
      <div class="intent-badge" v-if="role === 'assistant' && intent" :class="'intent-' + intent">
        {{ intentLabel }}
      </div>
      <div v-if="role === 'user'">{{ content }}</div>
      <div v-else v-html="rendered"></div>
    </div>
  </div>
</template>

<script>
import { marked } from 'marked'
export default {
  props: { role: String, content: String, streaming: Boolean, intent: String },
  computed: {
    rendered() {
      return marked(this.content || '', { breaks: true, gfm: true })
    },
    intentLabel() {
      const map = { kb: '知识库', chat: '闲聊', web: '联网', clarify: '追问' }
      return map[this.intent] || this.intent
    },
  },
}
</script>
