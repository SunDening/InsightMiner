<template>
  <div>
    <div class="upload-zone"
      :class="{ dragover: dragging, 'has-error': uploadError }"
      @dragover.prevent="dragging = true"
      @dragleave="dragging = false; uploadError = ''"
      @drop.prevent="handleDrop"
      @click="uploadError = ''; $refs.input.click()">
      <div class="icon">⌗</div>
      <div class="hint" v-if="!uploading">拖拽文档到此处，或点击选择</div>
      <div class="hint" v-else>正在上传 ({{ uploaded }}/{{ total }})...</div>
      <div class="sub-hint">支持 .txt .pdf .docx .csv .md 格式，可多选</div>
      <input ref="input" type="file" hidden multiple
        accept=".txt,.md,.pdf,.docx,.csv,.json,.yaml"
        @change="handleFiles" />
    </div>
    <div v-if="uploadError" class="upload-error">{{ uploadError }}</div>
  </div>
</template>

<script>
export default {
  props: {
    uploadFn: { type: Function, required: true },
  },
  data: () => ({ dragging: false, uploading: false, uploaded: 0, total: 0, uploadError: '' }),
  methods: {
    async handleDrop(e) {
      this.dragging = false
      this.uploadError = ''
      const files = Array.from(e.dataTransfer.files)
      if (files.length) await this.uploadAll(files)
    },
    async handleFiles(e) {
      this.uploadError = ''
      const files = Array.from(e.target.files)
      if (files.length) await this.uploadAll(files)
      e.target.value = ''
    },
    async uploadAll(files) {
      this.uploading = true
      this.uploaded = 0
      this.total = files.length
      let firstError = ''
      for (const file of files) {
        try {
          await this.uploadFn(file)
          this.uploaded++
        } catch (e) {
          // Continue uploading remaining files; show the first error
          if (!firstError) firstError = e.message || '上传失败'
          this.uploaded++
        }
      }
      this.uploading = false
      if (firstError) {
        this.uploadError = firstError
      }
    },
  },
}
</script>
