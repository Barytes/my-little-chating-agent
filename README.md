# AI Agent Chat

一个基于 FastAPI 的 AI Agent 聊天应用，支持工具调用（网络搜索、网页阅读）。

## 功能

- **Agent Loop**: 自动处理 LLM 的工具调用，最多循环 5 次
- **Web Search**: 使用 Tavily 搜索引擎搜索网络信息
- **Read Page**: 获取网页内容并提取纯文本
- **Chat Web UI**: 现代化的聊天界面，支持历史记录和实时响应
- **Agentic RAG**: 在前端选择本地 Markdown 笔记目录，自动生成/复用本地 FAISS index，并让 Agent 查询个人知识库
- **Strategic Information Radar**: 一键运行两阶段外部新闻扫描，并按战略背景评估重要性

## 项目结构

```
client.py        - AI Builders Space API 配置和客户端
tools.py         - 工具函数定义（web_search, read_page, run_scan）和 Function Schema
radar.py         - Strategic Information Radar 两阶段扫描流程
background.md    - 战略背景配置
indexer.py       - Markdown 笔记索引脚本（embeddings + FAISS）
main.py          - FastAPI 后端服务
static/index.html - 聊天前端界面
.env             - 环境变量配置（SUPER_MIND_API_KEY / BUILDER_API_KEY）
index/           - 本地笔记索引目录（自动生成，已 gitignore）
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 聊天前端界面 |
| `/models` | GET | 获取可用模型列表 |
| `/chat` | POST | Agent 聊天（自动处理工具调用） |
| `/run-scan` | POST | 运行 Strategic Information Radar 两阶段扫描 |
| `/run-scan/stream` | GET | 通过 SSE 实时流式返回扫描 trace 和最终结果 |
| `/search` | POST | 网络搜索 |
| `/notes/status` | GET | 查看当前加载的笔记索引 |
| `/notes/select` | POST | 切换到已有笔记索引 |
| `/notes/index` | POST | 上传所选笔记目录并生成索引 |

## 快速开始

### 1. 安装依赖

```bash
uv pip install fastapi uvicorn openai python-dotenv httpx beautifulsoup4 lxml
```

### 2. 配置环境变量

创建 `.env` 文件：

```
SUPER_MIND_API_KEY=your_api_key_here
RADAR_MODEL=grok-4-fast
```

`BUILDER_API_KEY` 和 `AI_BUILDER_TOKEN` 仍作为兼容 fallback 支持。
`RADAR_MODEL` 控制 Strategic Radar 使用的模型；如果写成 `grok4fast`，后端会自动映射为 `grok-4-fast`。

### 3. 启动服务

```bash
source .venv/bin/activate && uvicorn main:app --reload
```

### 4. 访问应用

打开浏览器访问: http://localhost:8000

页面里的 `Strategic Radar` 面板支持 `Start Scan` 一键扫描。扫描时会通过 SSE 显示 Live Trace，包括阶段切换、模型调用摘要、工具调用参数和工具结果摘要。

### 5. 加载个人笔记

在页面右上角点击 `Choose Notes`，选择包含 `.md` 文件的本地目录。

- 首次选择某个目录名时，会生成索引到 `index/<目录名>/`
- 再次选择同名目录时，会直接加载已有索引
- Agent 的 `query_my_notes` 工具会自动使用当前 active index

## 示例对话

- "Who won the Super Bowl 2025?" → 自动搜索网络并返回答案
- "Search for the latest release of Python, then read the official changelog page to tell me the new features." → 搜索 + 读网页 + 总结

## 技术栈

- **Backend**: FastAPI + OpenAI SDK
- **Frontend**: HTML/CSS/JavaScript (原生)
- **AI API**: AI Builders Space (https://space.ai-builders.com/backend/v1)
- **Tools**: Tavily Search, BeautifulSoup
