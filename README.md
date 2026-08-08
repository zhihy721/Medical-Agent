# Medical Agent

Medical Agent 是一个本地网页版智能问诊 Agent 原型。它不是医疗诊断系统，重点展示多轮问诊中的状态管理、风险分层、动态追问、工具调用和 LLM 接入。

当前默认使用 Flask 提供网页界面，Agent 编排默认走 LangGraph；没有配置模型 API 时也可以使用 Mock 演示模式体验流程。

## 快速开始

### Windows 一键安装

双击运行：

```bat
install.bat
```

安装脚本会创建 `.venv` 虚拟环境，并安装 `requirements.txt` 中的依赖。

### 启动网页

双击运行：

```bat
start.bat
```

或在命令行运行：

```bash
python start.py
```

启动后浏览器会打开：

```text
http://localhost:5000
```

如果浏览器没有自动打开，可以手动访问上面的地址。

## 首次配置 API

第一次启动时，如果项目根目录还没有 `config.env` 或没有 API Key，网页会先显示配置界面。

你可以选择：

- `DeepSeek`：填写 DeepSeek API URL、API Key、模型名等配置
- `Mock 演示模式`：不调用外部模型，只体验本地流程

点击“保存并开始”后，配置会写入本地 `config.env`。API Key 不会在网页里明文回显。

## 常用命令

启动网页但不自动打开浏览器：

```bash
python start.py --no-browser
```

指定端口启动：

```bash
python start.py --port 5050
```

运行系统检查：

```bash
python test_system.py
```

命令行演示模式：

```bash
python main.py
```

## 配置文件

网页配置会生成：

```text
config.env
```

示例字段：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_MAX_TOKENS=512
DEEPSEEK_TEMPERATURE=0.2
DATA_DIR=data
LOG_DIR=logs
```

`config.env` 已加入 `.gitignore`，不要提交真实 API Key。

## 数据与日志

运行时会自动生成两个本地目录（均已加入 `.gitignore`）：

```text
data/profiles/{user_id}.json     长期用户画像（JSON 文件持久化）
data/sessions/{session_id}.json  会话状态与对话历史
logs/app.log                     统一日志（滚动文件）
logs/events.jsonl                结构化事件流（执行轨迹）
```

服务重启后长期画像与会话状态不丢失；网页侧栏的“执行轨迹”区块和 `/api/debug/trace` 接口读取的就是事件流数据。

## 项目结构

```text
app.py                 Flask 网页入口
start.py               本地网页启动器
config_manager.py      本地配置读写和脱敏
templates/index.html   配置页、问诊页和运行状态侧栏
agent/                 Agent 编排、路由和控制器
llm/                   LLM 调用封装（含埋点与指标累计）
memory/                会话记忆、用户画像与 JSON 文件持久化
tools/                 症状、风险、指南工具，及工具协议与注册表
knowledge/             中医知识结构
observability/         日志、JSONL 事件流与指标汇总
test_system.py         系统检查脚本
```

## 当前能力

- 首次运行网页配置模型 API
- 保存配置后无需重启即可使用新配置
- 支持 DeepSeek 和 Mock 演示模式
- 多轮问诊状态管理
- 症状抽取、缺失字段追问、风险等级展示
- 长期用户画像和短期会话状态分离，JSON 文件持久化，重启不丢失
- LangGraph 作为默认 Agent runtime
- 统一工具协议（ToolResult + 版本 + 异常降级）与工具注册表
- 可观测性：统一日志、结构化事件流、LLM 耗时/token 指标、网页侧栏执行轨迹
- 调试接口 `/api/debug/trace`：查看当前会话的节点耗时、风险结果与 replan 记录

## 注意

本项目仅用于工程原型和演示，不替代医生面诊、诊断或治疗建议。高风险症状应及时就医。
