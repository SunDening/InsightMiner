import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { getMe, login, loginWithCode, register } from '@/services/api'

export const useAuthStore = defineStore('auth', () => {
  const user = ref(JSON.parse(localStorage.getItem('user') || 'null'))
  const token = ref(localStorage.getItem('token') || '')

  const isLoggedIn = computed(() => !!token.value && !!user.value)

  function _save(userData, accessToken) {
    user.value = userData
    token.value = accessToken
    localStorage.setItem('user', JSON.stringify(userData))
    localStorage.setItem('token', accessToken)
  }

  async function doRegister(data) {
    const res = await register(data)
    _save(res.user, res.access_token)
    return res
  }

  async function doLogin(data) {
    const res = await login(data)
    _save(res.user, res.access_token)
    return res
  }

  async function doLoginWithCode(data) {
    const res = await loginWithCode(data)
    _save(res.user, res.access_token)
    return res
  }

  async function fetchMe() {
    try {
      const u = await getMe()
      user.value = u
      localStorage.setItem('user', JSON.stringify(u))
      return u
    } catch {
      logout()
      return null
    }
  }

  function logout() {
    user.value = null
    token.value = ''
    localStorage.removeItem('user')
    localStorage.removeItem('token')
  }

  return {
    user,
    token,
    isLoggedIn,
    doRegister,
    doLogin,
    doLoginWithCode,
    fetchMe,
    logout,
  }
})
