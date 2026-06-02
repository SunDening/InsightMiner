<template>
  <div class="sidebar-section">
    <span>知识库</span>
  </div>
  <div class="sidebar-list">
    <div class="sidebar-item" v-for="kb in list" :key="kb.kb_id"
      :class="{ active: kb.kb_id === active }"
      @click="$emit('select', kb.kb_id)">
      <span class="item-icon">◆</span>
      <span class="item-label">{{ kb.kb_id }}</span>
      <span class="badge">{{ kb.document_count }}</span>
      <button class="del-btn" @click.stop="$emit('delete', kb.kb_id)" title="删除知识库">×</button>
    </div>
  </div>
  <template v-if="!creating">
    <button class="sidebar-action" @click="creating = true">＋ 新建知识库</button>
  </template>
  <template v-else>
    <div class="inline-form">
      <input ref="input" v-model="name" placeholder="知识库名称"
        @keydown.enter="confirm" @keydown.esc="cancel" />
      <button class="confirm-btn" @click="confirm">确定</button>
      <button class="cancel-btn" @click="cancel">取消</button>
    </div>
  </template>
</template>

<script>
import { ref, nextTick, watch } from 'vue'
export default {
  props: { list: Array, active: String },
  emits: ['select', 'create', 'delete'],
  setup(_, { emit }) {
    const creating = ref(false)
    const name = ref('')
    const input = ref(null)

    watch(creating, async (val) => {
      if (val) {
        await nextTick()
        input.value?.focus()
      }
    })

    async function confirm() {
      const v = name.value.trim()
      if (v) {
        emit('create', v)
        name.value = ''
        creating.value = false
      }
    }
    function cancel() {
      name.value = ''
      creating.value = false
    }

    return { creating, name, input, confirm, cancel }
  },
}
</script>
