# InsightMiner 未来扩展计划

## Phase 2 — 流式输出

- [ ] LLM 调用改造为 stream=True，逐 token 推送
- [ ] SSE 事件推送机制（event: status / token / evidence / done）
- [ ] 前端 StreamingText 组件对接

## Phase 3 — 前端完整实现

- [ ] Vue 3 三栏布局（知识库列表 | 聊天区域 | 证据面板）
- [ ] ChatView + 流式文本渲染
- [ ] EvidencePanel（置信度百分比进度条 + 原文展示）
- [ ] KnowledgeBase 管理页（上传进度、文档状态）
- [ ] 对话管理（新建/切换/删除，自动标题摘要）

## Phase 4 — 增强特性

- [ ] LLM 提供商运行时切换（DeepSeek / Ollama），无需重启
- [ ] 问题建议（文档上传后自动生成示例问题）
- [ ] 置信度阈值可调
- [ ] 源文档快速跳转
- [ ] 多知识库隔离（用户可创建多个独立知识库）

## 待评估

- [ ] 用户注册登录体系
- [ ] 多用户数据隔离
- [ ] 文档级权限控制
- [ ] 图片/扫描件 OCR 支持
