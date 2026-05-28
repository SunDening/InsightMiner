<template>
  <div v-if="evidences.length">
    <div class="evidence-header">
      <h3>引用证据</h3>
      <span class="count">{{ evidences.length }} 条</span>
    </div>
    <div class="evidence-cards">
      <div class="evidence-card" v-for="(ev, i) in evidences" :key="i">
        <div class="source">
          <span class="doc-icon">◈</span>
          <strong>{{ ev.source_document || '未知来源' }}</strong>
        </div>
        <div class="conf-bar">
          <div class="conf-bar-fill" :style="{ width: ev.confidence_pct + '%' }"></div>
        </div>
        <div class="conf-label">
          <span>置信度</span>
          <span>{{ ev.confidence_pct }}%</span>
        </div>
        <div class="content">{{ ev.content }}</div>
      </div>
    </div>
  </div>
  <div v-else class="evidence-empty">
    发送问题后，检索到的证据将在下方展示
  </div>
</template>

<script>
export default { props: { evidences: { type: Array, default: () => [] } } }
</script>
