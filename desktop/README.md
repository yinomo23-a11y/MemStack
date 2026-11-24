# 忆栈 MemStack 桌面端

Windows 10 / 11 x64。安装后从开始菜单打开「忆栈 MemStack」。需要 Microsoft Edge WebView2 Runtime。

## 接入数据

- **忆栈目录**：选择保存忆栈池文件的目录。公开版支持 `active.json`、`shortterm.json`、`cold.json` 和 `working_mem.json`。五池版也支持 `owner_*.json` 和私密池文件。私密池内容和标签在界面中隐藏。程序只读取池文件。
- **SQLite / JSONL**：选择文件。SQLite 需选择数据表，两种文件都需要确认内容字段。强度、时间、分组和实体标签可选。时间支持 ISO 8601 字符串、Unix 秒与毫秒。标签支持 JSON 字符串数组、分隔文本和带 `name` 字段的对象数组。
- **HTTP**：输入服务基础地址，可含 `/api` 等路径前缀。填写的是 HTTP 服务地址。数据库账号和密码由服务管理。远程地址建议使用 HTTPS。
- **合成演示**：在未接入数据时可直接打开。切换演示后，重新打开 App 会恢复上次保存的数据连接。

App 本地保存连接位置及字段映射。连接数据源后每 30 秒重新读取，也可点击「刷新快照」。读取失败时保留上次快照，每 5 秒重试。安装后可用 `memstack://reconnect` 打开 App 并立即重新连接本机保存的数据源。图表在内存中生成。一份快照最多显示 20,000 条记录，截取时会在界面标明。WebGL 不可用时可用流转矩阵。

## 六个内置模块

01 星图、02 流转矩阵、03 全景分屏沿用原版绘制布局，从当前连接的数据生成。04 的衰减推演和召回重排、05 的预算与交接预演、06 的摘要和校验都在 App 里计算。选目录或文件即可使用全部页面。

忆栈目录中的模型参数、召回审计、交接和蒸馏记录可直接读取。连接目录后，可在「连接数据」中关联 DSH 会话目录与 Vault 笔记库。05 读取 DSH 会话档中的实测 token，06 展示已有蒸馏卡片、Vault 状态和源文件指纹。

兼容的本机忆栈模型可在连接设置中启用写入。04 可保存模型参数，05 可保存交接单与关联的 DSH 配置，06 可保存本机规则生成的候选、审核卡片、AI 自动分批预审（仅出标记与理由，用户可逐条改判后确认原子落盘）、校准证据并记录技能结果。蒸馏试跑和 Vault 护栏测试保留预演模式。SQLite、JSONL、HTTP 和演示模式使用本机预演。

04、05、06 的「回滚本页」恢复本页最近一次实际写入。顶部「撤销写入」恢复当前数据源最近一次实际写入。App 为每个连接保留最多 10 步撤销记录。目标文件有其他改动时，App 会停止撤销并保留备份。

## HTTP 协议 v1

请求均为 GET。接口返回 UTF-8 JSON。必需的两个端点如下。

```text
GET /capabilities
GET /snapshot
```

```json
{"version":1,"features":["snapshot","audit"],"pools":[{"id":"active","label":"活跃记忆"}],"strength_unit":"分"}
```

```json
{"records":[{"content":"合成的记忆条目","strength":72,"time":"2026-01-01T12:00:00Z","pool":"active","entities":["示例主题"]}]}
```

`content` 为正文。其他字段可省略，也可用 `private: true` 隐藏正文和标签。可选 `truncated: true` 表明服务截取了记录。分组名称取自能力声明，未声明的分组使用其字段值。

可选的 `/telemetry`、`/settings`、`/handover`、`/cards`、`/audit` 需在 `features` 中声明。返回对象包含 `rows` 数组，每行含 `label`、`value` 与可选的 `section` 字符串。衰减、交接、蒸馏及审计页面只读展示服务提供的数据。附加分区如下。

| 端点 | 默认分区 | 可选分区 |
| --- | --- | --- |
| telemetry | 召回活跃度 | decay、ranking、budget |
| settings | 衰减与重排参数 | runtime |
| handover | 交接单 | preview |
| cards | 记忆蒸馏 | skills |
| audit | 安全审计 | hashes |

运行 `python adapter_example.py`，在 App 中连接 `http://127.0.0.1:8787` 验证接入。示例只在本机监听。

## 从源码构建

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe app.py
.\build.ps1
```

构建需要 Inno Setup 6。可通过 `-InnoCompiler` 指定 ISCC 路径。安装包输出到 `dist`。

