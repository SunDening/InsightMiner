import { createRouter, createWebHashHistory } from 'vue-router'
import ChatView from '@/views/ChatView.vue'
import KnowledgeBase from '@/views/KnowledgeBase.vue'
import LoginView from '@/views/Login.vue'
import RegisterView from '@/views/Register.vue'

const routes = [
  { path: '/login', name: 'login', component: LoginView, meta: { layout: 'auth' } },
  { path: '/register', name: 'register', component: RegisterView, meta: { layout: 'auth' } },
  { path: '/', name: 'chat', component: ChatView, meta: { layout: 'app' } },
  { path: '/knowledge-base', name: 'kb', component: KnowledgeBase, meta: { layout: 'app' } },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

// Navigation guard — redirect to /login if not authenticated
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  const publicPages = ['/login', '/register']
  if (!token && !publicPages.includes(to.path)) {
    next('/login')
  } else {
    next()
  }
})

export default router
