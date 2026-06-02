import { createRouter, createWebHashHistory } from 'vue-router'
import ChatView from '@/views/ChatView.vue'
import KnowledgeBase from '@/views/KnowledgeBase.vue'

const routes = [
  { path: '/', name: 'chat', component: ChatView },
  { path: '/knowledge-base', name: 'kb', component: KnowledgeBase },
]

export default createRouter({
  history: createWebHashHistory(),
  routes,
})
