<template>
  <div class="auth-page">
    <!-- Brand watermark -->
    <div class="brand-watermark">IM</div>

    <!-- Dot grid pattern overlay -->
    <div class="dot-grid"></div>

    <!-- Decorative background orbs -->
    <div class="bg-orbs">
      <div class="orb orb-1"></div>
      <div class="orb orb-2"></div>
      <div class="orb orb-3"></div>
    </div>

    <!-- Geometric ring decorations -->
    <div class="ring ring-1"></div>
    <div class="ring ring-2"></div>

    <!-- Floating symbols -->
    <div class="float-symbol symbol-diamond" style="top:12%;left:8%">◆</div>
    <div class="float-symbol symbol-plus" style="top:25%;right:12%">+</div>
    <div class="float-symbol symbol-ring" style="bottom:20%;left:5%">○</div>
    <div class="float-symbol symbol-cross" style="bottom:30%;right:8%">✧</div>
    <div class="float-symbol symbol-dot" style="top:45%;right:4%">•</div>
    <div class="float-symbol symbol-dot" style="top:8%;left:45%">•</div>

    <div class="auth-card">
      <!-- Accent bar at top of card -->
      <div class="card-accent"></div>
      <!-- Corner decoration -->
      <div class="card-corner"></div>

      <div class="auth-header">
        <div class="auth-logo">IM</div>
        <h1>注册</h1>
        <p class="auth-subtitle">创建你的 InsightMiner 账号</p>
      </div>

      <form class="auth-form" @submit.prevent="handleRegister">
        <div class="field">
          <label>用户名</label>
          <div class="input-wrap">
            <span class="input-icon">👤</span>
            <input v-model="form.username" type="text" placeholder="2-32 个字符" required />
          </div>
        </div>
        <div class="field">
          <label>邮箱</label>
          <div class="input-wrap">
            <span class="input-icon">✉</span>
            <input v-model="form.email" type="email" placeholder="your@email.com" required />
          </div>
        </div>
        <div class="field">
          <label>密码</label>
          <div class="input-wrap">
            <span class="input-icon">🔒</span>
            <input v-model="form.password" type="password" placeholder="至少 6 位" required />
          </div>
        </div>
        <div class="field">
          <label>确认密码</label>
          <div class="input-wrap">
            <span class="input-icon">🔒</span>
            <input v-model="form.confirm_password" type="password" placeholder="再次输入密码" required />
          </div>
        </div>
        <p v-if="error" class="form-error">{{ error }}</p>
        <button type="submit" class="submit-btn" :disabled="loading">
          <span v-if="loading" class="spinner"></span>
          <span v-else>注册</span>
        </button>
      </form>

      <div class="auth-footer">
        已有账号？
        <router-link to="/login">立即登录</router-link>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

export default {
  name: 'RegisterView',
  setup() {
    const router = useRouter()
    const auth = useAuthStore()
    const loading = ref(false)
    const error = ref('')
    const form = ref({
      username: '',
      email: '',
      password: '',
      confirm_password: '',
    })

    async function handleRegister() {
      error.value = ''
      if (form.value.password !== form.value.confirm_password) {
        error.value = '两次输入的密码不一致'
        return
      }
      if (form.value.password.length < 6) {
        error.value = '密码长度不能少于 6 位'
        return
      }
      loading.value = true
      try {
        await auth.doRegister(form.value)
        router.push('/')
      } catch (e) {
        error.value = e?.response?.data?.detail?.[0]?.msg || e?.response?.data?.detail || '注册失败'
      } finally {
        loading.value = false
      }
    }

    return { form, loading, error, handleRegister }
  },
}
</script>

<style scoped>
/* ════════════════════════════════════════════════
   Auth Pages — Branded Design
   ════════════════════════════════════════════════ */

.auth-page {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: linear-gradient(135deg, var(--color-brand-navy) 0%, #1a1a4e 40%, var(--color-primary-deep) 100%);
  padding: var(--space-lg);
  overflow: hidden;
}

/* ── Brand Watermark ── */

.brand-watermark {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: clamp(300px, 50vw, 600px);
  font-weight: 900;
  color: rgba(255, 255, 255, 0.03);
  letter-spacing: -40px;
  pointer-events: none;
  user-select: none;
  z-index: 0;
  white-space: nowrap;
  line-height: 1;
}

/* ── Dot Grid Pattern ── */

.dot-grid {
  position: absolute;
  inset: 0;
  background-image: radial-gradient(circle, rgba(255, 255, 255, 0.07) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  z-index: 0;
}

/* ── Background Orbs ── */

.bg-orbs {
  position: absolute;
  inset: 0;
  pointer-events: none;
  overflow: hidden;
  z-index: 0;
}

.orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(100px);
  opacity: 0.12;
  animation: orbFloat 25s ease-in-out infinite;
}

.orb-1 {
  width: 600px;
  height: 600px;
  background: var(--color-primary);
  top: -20%;
  right: -15%;
  animation-delay: 0s;
}

.orb-2 {
  width: 500px;
  height: 500px;
  background: var(--color-brand-teal);
  bottom: -20%;
  left: -15%;
  animation-delay: -8s;
}

.orb-3 {
  width: 350px;
  height: 350px;
  background: var(--color-brand-purple);
  top: 40%;
  right: 30%;
  animation-delay: -16s;
}

@keyframes orbFloat {
  0%, 100% { transform: translate(0, 0) scale(1); }
  25% { transform: translate(40px, -50px) scale(1.1); }
  50% { transform: translate(-30px, 30px) scale(0.95); }
  75% { transform: translate(50px, 40px) scale(1.05); }
}

/* ── Geometric Rings ── */

.ring {
  position: absolute;
  border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.06);
  pointer-events: none;
  z-index: 0;
  animation: ringPulse 8s ease-in-out infinite;
}

.ring-1 {
  width: 400px;
  height: 400px;
  top: 15%;
  right: 5%;
}

.ring-2 {
  width: 250px;
  height: 250px;
  bottom: 25%;
  left: 10%;
  animation-delay: -4s;
}

@keyframes ringPulse {
  0%, 100% { transform: scale(1); opacity: 0.06; }
  50% { transform: scale(1.3); opacity: 0.02; }
}

/* ── Floating Symbols ── */

.float-symbol {
  position: absolute;
  color: rgba(255, 255, 255, 0.08);
  pointer-events: none;
  z-index: 0;
  animation: symbolDrift 12s ease-in-out infinite;
}

.symbol-diamond { font-size: 20px; animation-delay: 0s; }
.symbol-plus    { font-size: 28px; font-weight: 100; animation-delay: -3s; }
.symbol-ring    { font-size: 18px; animation-delay: -6s; }
.symbol-cross   { font-size: 22px; animation-delay: -9s; }
.symbol-dot     { font-size: 32px; animation-delay: -2s; }

@keyframes symbolDrift {
  0%, 100% { transform: translateY(0) rotate(0deg); opacity: 0.08; }
  50% { transform: translateY(-20px) rotate(180deg); opacity: 0.15; }
}

/* ── Card Decorative Elements ── */

.card-accent {
  position: absolute;
  top: -1px;
  left: 50%;
  transform: translateX(-50%);
  width: 60%;
  height: 4px;
  background: linear-gradient(90deg, transparent, var(--color-primary), var(--color-brand-teal), transparent);
  border-radius: var(--radius-full);
  opacity: 0.7;
  z-index: 3;
}

.card-corner {
  position: absolute;
  top: -8px;
  right: -8px;
  width: 48px;
  height: 48px;
  border-top: 3px solid rgba(86, 69, 212, 0.2);
  border-right: 3px solid rgba(86, 69, 212, 0.2);
  border-radius: 0 var(--radius-lg) 0 0;
  z-index: 3;
}

/* ── Card ── */

.auth-card {
  position: relative;
  width: 420px;
  max-width: 100%;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(24px) saturate(1.4);
  -webkit-backdrop-filter: blur(24px) saturate(1.4);
  border-radius: var(--radius-xxl);
  padding: var(--space-xxxl);
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.15),
    0 24px 80px rgba(0, 0, 0, 0.35),
    0 8px 32px rgba(0, 0, 0, 0.2);
  z-index: 2;
  animation: cardIn 0.5s cubic-bezier(0.16, 1, 0.3, 1);
  overflow: visible;
}

@keyframes cardIn {
  from {
    opacity: 0;
    transform: translateY(24px) scale(0.97);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

/* ── Header ── */

.auth-header {
  text-align: center;
  margin-bottom: var(--space-xxl);
}

.auth-logo {
  width: 52px;
  height: 52px;
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-deep) 100%);
  border-radius: var(--radius-lg);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  font-weight: 700;
  color: var(--color-on-primary);
  margin: 0 auto var(--space-md);
  box-shadow: 0 4px 12px rgba(86, 69, 212, 0.3);
}

.auth-header h1 {
  font-size: var(--fs-h4);
  font-weight: 700;
  color: var(--color-ink);
  letter-spacing: -0.5px;
  margin-bottom: 4px;
}

.auth-subtitle {
  font-size: var(--fs-body-sm);
  color: var(--color-steel);
}

/* ── Form ── */

.auth-form {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.field label {
  font-size: var(--fs-caption);
  font-weight: 600;
  color: var(--color-slate);
  letter-spacing: 0.3px;
}

/* ── Input with icon ── */

.input-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 12px;
  border: 1.5px solid var(--color-hairline);
  border-radius: var(--radius-md);
  background: var(--color-canvas);
  transition: border-color var(--ease), box-shadow var(--ease);
}

.input-wrap:focus-within {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(86, 69, 212, 0.1);
}

.input-icon {
  font-size: 14px;
  opacity: 0.5;
  flex-shrink: 0;
}

.input-wrap input {
  flex: 1;
  padding: 11px 0;
  border: none;
  font-size: var(--fs-body-sm);
  outline: none;
  background: transparent;
  color: var(--color-charcoal);
}

.input-wrap input::placeholder {
  color: var(--color-muted);
}

/* ── Error ── */

.form-error {
  font-size: var(--fs-body-sm);
  color: var(--color-error);
  background: rgba(224, 49, 49, 0.07);
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  border-left: 3px solid var(--color-error);
}

/* ── Submit ── */

.submit-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 13px;
  border-radius: var(--radius-md);
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-deep) 100%);
  color: var(--color-on-primary);
  font-size: var(--fs-body);
  font-weight: 600;
  transition: opacity var(--ease), transform var(--ease);
  margin-top: var(--space-xs);
  border: none;
  cursor: pointer;
  letter-spacing: 0.3px;
  box-shadow: 0 4px 14px rgba(86, 69, 212, 0.25);
}

.submit-btn:hover:not(:disabled) {
  opacity: 0.92;
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(86, 69, 212, 0.35);
}

.submit-btn:active:not(:disabled) {
  transform: translateY(0);
}

.submit-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  box-shadow: none;
  transform: none;
}

/* ── Spinner ── */

.spinner {
  display: inline-block;
  width: 18px;
  height: 18px;
  border: 2.5px solid rgba(255, 255, 255, 0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* ── Footer ── */

.auth-footer {
  text-align: center;
  margin-top: var(--space-xl);
  padding-top: var(--space-lg);
  border-top: 1px solid var(--color-hairline);
  font-size: var(--fs-body-sm);
  color: var(--color-steel);
}

.auth-footer a {
  font-weight: 600;
  color: var(--color-primary);
  transition: color var(--ease);
}

.auth-footer a:hover {
  color: var(--color-primary-pressed);
}
</style>
