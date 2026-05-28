<template>
  <div class="upload-zone"
    :class="{ dragover: dragging }"
    @dragover.prevent="dragging = true"
    @dragleave="dragging = false"
    @drop.prevent="handleDrop"
    @click="$refs.input.click()">
    <div class="icon">⌗</div>
    <div class="hint" v-if="!uploading">拖拽文档到此处，或点击选择</div>
    <div class="hint" v-else>正在上传 ({{ uploaded }}/{{ total }})...</div>
    <div class="sub-hint">支持 .txt .pdf .docx .csv .md 格式，可多选</div>
    <input ref="input" type="file" hidden multiple
      accept=".txt,.md,.pdf,.docx,.csv,.json,.yaml"
      @change="handleFiles" />
  </div>
</template>

<script>
export default {
  props: {
    uploadFn: { type: Function, required: true },
  },
  data: () => ({ dragging: false, uploading: false, uploaded: 0, total: 0 }),
  methods: {
    async handleDrop(e) {
      this.dragging = false
      const files = Array.from(e.dataTransfer.files)
      if (files.length) await this.uploadAll(files)
    },
    async handleFiles(e) {
      const files = Array.from(e.target.files)
      if (files.length) await this.uploadAll(files)
      e.target.value = ''
    },
    async uploadAll(files) {
      this.uploading = true
      this.uploaded = 0
      this.total = files.length
      for (const file of files) {
        await this.uploadFn(file)
        this.uploaded++
      }
      this.uploading = false
    },
  },
}
</script>
