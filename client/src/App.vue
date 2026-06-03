<template>
  <!-- ── Auth layout (login/register): full screen, no sidebar ── -->
  <template v-if="isAuthPage">
    <router-view />
  </template>

  <!-- ── App layout (authenticated pages): sidebar + content ── -->
  <div v-else class="app-layout">
    <aside class="sidebar">
      <div class="sidebar-header">
        <div class="logo-mark">IM</div>
        <span class="brand-name">InsightMiner</span>
      </div>

      <!-- User info / login status -->
      <div class="sidebar-user">
        <template v-if="auth.isLoggedIn">
          <div class="user-avatar">{{ auth.user.username.charAt(0).toUpperCase() }}</div>
          <div class="user-name">{{ auth.user.username }}</div>
          <button class="logout-btn" @click="handleLogout" title="退出登录">↩</button>
        </template>
        <template v-else>
          <router-link to="/login" class="login-link">登录 / 注册</router-link>
        </template>
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
import { computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useChatStore } from '@/stores/chat'
import { useAuthStore } from '@/stores/auth'
import KnowledgeBaseList from '@/components/KnowledgeBaseList.vue'
import ConversationList from '@/components/ConversationList.vue'
import { createKB, deleteKB } from '@/services/api'

export default {
  components: { KnowledgeBaseList, ConversationList },
  setup() {
    const router = useRouter()
    const route = useRoute()
    const store = useChatStore()
    const auth = useAuthStore()

    const isAuthPage = computed(() => route.meta?.layout === 'auth')

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

    function handleLogout() {
      auth.logout()
      router.push('/login')
    }

    onMounted(async () => {
      // Only load app data if this is not an auth page
      if (isAuthPage.value) return

      // Verify token is still valid on mount
      if (auth.isLoggedIn) {
        const u = await auth.fetchMe()
        if (!u) {
          router.push('/login')
          return
        }
      }
      await store.fetchKBs()
      await store.fetchDocs()
      await store.fetchThreads()
    })

    return { store, auth, isAuthPage, switchKb, createKb, deleteKb, handleLogout }
  },
}
</script>

<style scoped>
.sidebar-user {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: var(--space-sm) var(--space-md);
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  flex-shrink: 0;
  min-height: 44px;
}

.user-avatar {
  width: 26px;
  height: 26px;
  border-radius: var(--radius-full);
  background: var(--color-primary);
  color: var(--color-on-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 700;
  flex-shrink: 0;
}

.user-name {
  flex: 1;
  font-size: var(--fs-body-sm);
  font-weight: 600;
  color: var(--color-on-dark);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.logout-btn {
  background: none;
  border: none;
  color: var(--color-on-dark-dim);
  font-size: 16px;
  padding: 2px 6px;
  cursor: pointer;
  border-radius: var(--radius-sm);
  transition: color var(--ease), background var(--ease);
  line-height: 1;
}

.logout-btn:hover {
  color: var(--color-on-dark);
  background: rgba(255, 255, 255, 0.08);
}

.login-link {
  font-size: var(--fs-body-sm);
  font-weight: 500;
  color: var(--color-on-dark-muted);
  text-decoration: none;
  transition: color var(--ease);
  padding: 4px 0;
}

.login-link:hover {
  color: var(--color-on-dark);
}
</style>
