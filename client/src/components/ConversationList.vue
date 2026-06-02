<template>
  <div class="sidebar-section">
    <span>对话</span>
  </div>
  <div class="sidebar-item new-item" @click="$emit('new')">
    <span>＋</span>
    <span class="item-label">新建对话</span>
  </div>
  <div class="sidebar-list grow">
    <div class="sidebar-item" v-for="t in threads" :key="t.thread_id"
      :class="{ active: t.thread_id === active }"
      @click="$emit('select', t.thread_id)">
      <span class="item-icon">◌</span>
      <span class="item-label">{{ t.title || t.thread_id.slice(0, 10) + '...' }}</span>
      <button class="del-btn" @click.stop="$emit('delete', t.thread_id)" title="删除对话">×</button>
    </div>
    <div v-if="!threads.length" class="empty-hint">
      暂无对话
    </div>
  </div>
</template>

<script>
export default {
  props: { threads: Array, active: String },
  emits: ['select', 'new', 'delete'],
}
</script>

<style scoped>
.empty-hint {
  padding: var(--space-sm) var(--space-sm);
  font-size: var(--fs-body-sm);
  color: var(--color-on-dark-dim);
  text-align: center;
}
</style>
