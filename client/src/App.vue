<template>
  <div class="app-layout">
    <aside class="sidebar">
      <div class="sidebar-header">
        <div class="logo-mark">IM</div>
        <span class="brand-name">InsightMiner</span>
      </div>

      <nav class="sidebar-nav">
        <router-link to="/">
          <span class="nav-icon">⌂</span>
          对话
        </router-link>
        <router-link to="/knowledge-base">
          <span class="nav-icon">⊞</span>
          知识库
        </router-link>
      </nav>

      <KnowledgeBaseList
        :list="store.kbList"
        :active="store.activeKbId"
        @select="switchKb"
        @create="createKb"
        @delete="deleteKb" />

      <ConversationList
        :threads="store.threads"
        :active="store.activeThreadId"
        @select="store.switchThread"
        @new="store.newThread"
        @delete="store.removeThread" />
    </aside>

    <router-view />
  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useChatStore } from '@/stores/chat'
import KnowledgeBaseList from '@/components/KnowledgeBaseList.vue'
import ConversationList from '@/components/ConversationList.vue'
import { createKB, deleteKB } from '@/services/api'

export default {
  components: { KnowledgeBaseList, ConversationList },
  setup() {
    const store = useChatStore()

    async function switchKb(kbId) {
      await store.switchKb(kbId)
      store.newThread()
    }

    async function createKb(name) {
      await createKB(name)
      await store.fetchKBs()
    }

    async function deleteKb(kbId) {
      await deleteKB(kbId)
      if (store.activeKbId === kbId) {
        store.activeKbId = 'default'
        await store.switchKb('default')
        store.newThread()
      }
      await store.fetchKBs()
    }

    onMounted(async () => {
      await store.fetchKBs()
      await store.fetchDocs()
      await store.fetchThreads()
    })

    return { store, switchKb, createKb, deleteKb }
  },
}
</script>
