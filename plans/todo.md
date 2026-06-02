# InsightMiner Phase 3 — Vue 前端开发

## 目标

为 InsightMiner 实现完整的 Vue 3 前端，三栏布局，对接 SSE 流式接口。

## 最终项目结构

```
InsightMiner/                          (D:\Coding\Vue\code\InsightMiner)
├── vue.config.js                      # 代理 /api → 127.0.0.1:8765
├── src/
│   ├── main.js                        # 入口 (Pinia + Router)
│   ├── App.vue                        # 三栏布局 (侧边栏 | 对话 | 证据面板)
│   ├── assets/styles.css              # 全局样式
│   ├── router/index.js                # 路由: /, /knowledge-base, /settings
│   ├── stores/chat.js                 # Pinia Store (KB/对话/流式状态)
│   ├── services/api.js                # Axios + SSE 流式客户端
│   ├── views/
│   │   ├── ChatView.vue              # 主聊天视图 (消息列表 + 输入框)
│   │   ├── KnowledgeBase.vue         # 知识库管理 (上传/删除文档)
│   │   └── Settings.vue              # 系统设置 (LLM 提供商等)
│   └── components/
│       ├── MessageBubble.vue         # 消息气泡 (Markdown 渲染)
│       ├── EvidencePanel.vue         # 引用证据 (置信度进度条)
│       ├── KnowledgeBaseList.vue     # 知识库列表 (侧边栏)
│       ├── ConversationList.vue      # 对话列表 (侧边栏)
│       └── FileUploader.vue          # 拖拽上传组件
```

## 实现清单

### 1. 依赖与配置
- [x] 安装 vue-router@4, pinia, axios, marked
- [x] vue.config.js 配置 API 代理

### 2. 数据层
- [x] services/api.js — REST + SSE AsyncGenerator 客户端
- [x] stores/chat.js — Pinia store（KB/对话/流式三合一状态管理）

### 3. 布局与组件
- [x] 全局样式 (CSS Grid 三栏, 变量主题, 响应式)
- [x] 侧边栏组件 (KB 列表 + 对话列表 + 导航)
- [x] 消息气泡 (Markdown 渲染 via marked)
- [x] 证据面板 (置信度百分比进度条)
- [x] 文件上传组件 (拖拽 + 点击)

### 4. 视图
- [x] ChatView — 流式消息展示 + 输入框
- [x] KnowledgeBase — 文档上传/删除/状态
- [x] Settings — LLM 提供商选择

### 5. 验证
- [x] npm run build 通过 (0 error)
- [x] 代理联调成功 (/api/system/health → 200)
- [x] 构建产物 dist/ 就绪

## 启动方式

```bash
# 终端1: 启动后端 (端口 8765)
cd D:\Coding\agent\test\InsightMiner
uvicorn insight_miner.main:app --reload

# 终端2: 启动前端 (端口 8080)
cd D:\Coding\Vue\code\InsightMiner
npm run serve
```
