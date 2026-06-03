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

      <!-- Mind Map trigger (only for finished assistant messages) -->
      <div v-if="role === 'assistant' && !streaming && content" class="mindmap-actions">
        <button
          class="mindmap-btn"
          @click="toggleMindMap"
          :disabled="mindmapLoading"
        >
          <span v-if="mindmapLoading">⏳ 生成中...</span>
          <span v-else-if="showMindMap">✕ 收起导图</span>
          <span v-else>🧠 思维导图</span>
        </button>
        <button
          v-if="showMindMap"
          class="mindmap-btn mindmap-btn-download"
          @click="downloadMindMap"
          :disabled="mindmapDownloading"
        >
          {{ mindmapDownloading ? '⏳ 下载中...' : '⬇ 下载 PNG' }}
        </button>
      </div>

      <!-- Error message -->
      <div v-if="errorMsg" class="mindmap-error">{{ errorMsg }}</div>

      <!-- Mind Map SVG (must be <svg>, markmap treats it as the SVG root) -->
      <svg v-show="showMindMap" ref="mindmapRef" class="mindmap-container" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
  </div>
</template>

<script>
import { marked } from 'marked'
import { Markmap } from 'markmap-view'
import { toPng } from 'html-to-image'
import { generateMindmap } from '@/services/api'

export default {
  props: {
    role: String,
    content: String,
    streaming: Boolean,
    intent: String,
    threadId: String,
  },

  data() {
    return {
      showMindMap: false,
      mindmapLoading: false,
      mindmapDownloading: false,
      mindmapData: null,
      errorMsg: '',
      markmap: null,
    }
  },

  computed: {
    rendered() {
      return marked(this.content || '', { breaks: true, gfm: true })
    },
    intentLabel() {
      const map = { kb: '知识库', chat: '闲聊', web: '联网', clarify: '追问' }
      return map[this.intent] || this.intent
    },
  },

  methods: {
    async toggleMindMap() {
      if (this.showMindMap) {
        this.showMindMap = false
        return
      }

      if (this.mindmapData) {
        this.showMindMap = true
        await this.$nextTick()
        this.renderMindMap()
        return
      }

      this.mindmapLoading = true
      try {
        const data = await generateMindmap(this.threadId)
        this.mindmapData = data
        this.showMindMap = true
        await this.$nextTick()
        this.renderMindMap()
      } catch (err) {
        const detail = err?.response?.data?.detail || err.message || '生成失败'
        this.errorMsg = detail
        setTimeout(() => { this.errorMsg = '' }, 5000)
        this.mindmapLoading = false
        this.showMindMap = false
      } finally {
        this.mindmapLoading = false
      }
    },

    renderMindMap() {
      const el = this.$refs.mindmapRef
      if (!el) return
      // Clear previous content
      el.innerHTML = ''
      this.markmap = Markmap.create(el, {
        maxWidth: 400,
        colorFreezeLevel: 0,
        duration: 500,
        zoom: true,
        pan: true,
      }, this.mindmapData.root)
      // Fit after DOM settles
      this._fitTimer = setTimeout(() => this.markmap?.fit(), 100)
      this._fitHandler = () => this.markmap?.fit()
      window.addEventListener('resize', this._fitHandler)
    },
    async downloadMindMap() {
      const el = this.$refs.mindmapRef
      if (!el || this.mindmapDownloading) return
      this.mindmapDownloading = true
      try {
        const dataUrl = await toPng(el, {
          backgroundColor: '#ffffff',
          pixelRatio: 2,
          cacheBust: true,
        })
        const link = document.createElement('a')
        link.download = '思维导图.png'
        link.href = dataUrl
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
      } catch (err) {
        console.error('Download mind map failed:', err)
        this.errorMsg = '下载失败，请重试'
        setTimeout(() => { this.errorMsg = '' }, 5000)
      } finally {
        this.mindmapDownloading = false
      }
    },
  },
  beforeUnmount() {
    clearTimeout(this._fitTimer)
    if (this._fitHandler) {
      window.removeEventListener('resize', this._fitHandler)
    }
  },
}
</script>

<style scoped>
.mindmap-actions {
  margin-top: 14px;
}

.mindmap-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--color-surface, #f5f5f5);
  border: 1px solid var(--color-border, #e0e0e0);
  border-radius: 8px;
  padding: 6px 16px;
  font-size: 13px;
  font-family: inherit;
  cursor: pointer;
  color: var(--color-text-secondary, #555);
  transition: all 0.2s ease;
  line-height: 1.4;
}

.mindmap-btn:hover:not(:disabled) {
  background: #ede8ff;
  border-color: var(--color-primary, #5645d4);
  color: var(--color-primary, #5645d4);
}

.mindmap-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.mindmap-btn:active:not(:disabled) {
  transform: scale(0.97);
}

.mindmap-btn + .mindmap-btn {
  margin-left: 8px;
}

.mindmap-btn-download {
  background: #e8f5e9;
  border-color: #c8e6c9;
  color: #2e7d32;
}

.mindmap-btn-download:hover:not(:disabled) {
  background: #c8e6c9;
  border-color: #66bb6a;
  color: #1b5e20;
}

.mindmap-container {
  display: block;
  width: 100%;
  height: 420px;
  margin-top: 14px;
  border: 1px solid var(--color-border, #e0e0e0);
  border-radius: 10px;
  background: #ffffff;
  overflow: hidden;
}

.mindmap-container text {
  font-family: inherit;
  font-size: 13px;
}

.mindmap-error {
  margin-top: 8px;
  padding: 6px 12px;
  background: #fff0f0;
  border: 1px solid #ffd4d4;
  border-radius: 6px;
  color: #c00;
  font-size: 13px;
}
</style>
