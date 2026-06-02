<template>
  <div class="kb-page">
    <h1>知识库管理</h1>

    <FileUploader :upload-fn="handleUpload" />

    <div class="kb-toolbar">
      <select v-model="store.activeKbId" @change="switchKb">
        <option v-for="kb in store.kbList" :key="kb.kb_id" :value="kb.kb_id">{{ kb.kb_id }}</option>
      </select>
      <button class="btn" @click="fetchAll">↻ 刷新</button>
      <router-link to="/" class="btn-back">← 返回对话</router-link>
    </div>

    <div class="doc-list">
      <div class="doc-row" v-for="doc in store.docs" :key="doc.filename">
        <span class="doc-icon">◈</span>
        <span class="name">{{ doc.filename }}</span>
        <span class="size">{{ (doc.size_bytes / 1024).toFixed(1) }} KB</span>
        <span class="status" :class="doc.status">{{ doc.status }}</span>
        <button class="del" @click="remove(doc.filename)">×</button>
      </div>
      <div v-if="!store.docs.length" class="empty-docs">
        暂无文档，拖拽或点击上方区域上传
      </div>
    </div>
  </div>
</template>

<script>
import FileUploader from '@/components/FileUploader.vue'
import { useChatStore } from '@/stores/chat'

export default {
  components: { FileUploader },
  setup() {
    const store = useChatStore()

    async function fetchAll() {
      await store.fetchKBs()
      await store.fetchDocs()
    }

    async function handleUpload(file) {
      await store.uploadFile(file)
    }

    async function remove(filename) {
      await store.removeDoc(filename)
    }

    async function switchKb() {
      await store.switchKb(store.activeKbId)
    }

    return { store, handleUpload, remove, switchKb, fetchAll }
  },
}
</script>

<style scoped>
.empty-docs {
  text-align: center;
  padding: var(--space-xxxl);
  color: var(--color-stone);
  font-size: var(--fs-body-sm);
}
</style>
