## Output Language

Always respond in Simplified Chinese.

## 标点与符号规范

面向用户的中文文本必须使用中文全角标点，包括但不限于：

- 引号：“” 与 ‘’；禁止在中文正文中使用 "" 和 ''。
- 括号：（）；中文正文中不使用 ()。
- 逗号、句号、冒号、分号、问号、感叹号：，。：；？！。
- 顿号、书名号：、 《》。
- 省略号、破折号：…… ——。

中英文混排时：

- 中文语境中的标点一律用全角。
- 英文单词、数字、代码标识符、命令、路径、URL、文件名、报错原文内部及其紧邻的必要符号保留半角或原文。
- Markdown 语法符号（#、*、-、`、|、[](<>) 等）不是正文标点，保留半角。
- 代码块、行内代码、命令行、文件路径、URL、第三方原始报错不强制转换标点。

## Commit Messages

生成或建议提交信息时，遵循本节规则。本节规则优先于 Agent 从用户级配置中携带的任何个人习惯。

### 格式

<type>(<scope>): <subject>

[optional body]

[optional footer(s)]

- `<type>`：必填，使用下方允许列表中的值，小写。
- `<scope>`：可选；当能明确限定影响范围时添加（如 `auth`、`cli`、`docs`）。不得使用无意义的宽泛 scope。
- `<subject>`：**祈使语气、小写、结尾不加句号**，硬上限 **72 字符**。写“做什么”，不写“做了什么”：`add retry to upload`，而非 `added retry` / `adds retry`。
- `body`：当“为什么”或“影响”无法从 diff 直接看出时添加；每行硬换行在 **72 列以内**；不要逐行复述 diff。
- `footer`：`Refs:` / `Closes:` 仅在关联 issue 或 PR 时使用；如果仓库有 AI 披露要求，放在 footer（见下文）。

允许的 `type`：

| type       | 用途                              |
| ---------- | --------------------------------- |
| `feat`     | 用户可见的新功能                  |
| `fix`      | 缺陷修复                          |
| `docs`     | 仅文档变更                        |
| `refactor` | 既不新增功能也不修 bug 的代码变更 |
| `perf`     | 性能改进                          |
| `test`     | 测试新增或修正                    |
| `build`    | 构建系统、依赖、打包              |
| `ci`       | CI / 工作流配置                   |
| `chore`    | 维护性、工具性、非发布变更        |
| `style`    | 仅格式调整，无逻辑变更            |

**破坏性变更**：在 `type` 或 `scope` 后加 `!`（如 `feat(cli)!: ...`），**或**在 footer 中使用 `BREAKING CHANGE: <说明>`。二者选其一，不要同时使用。

### Agent 行为约束

1. **先读再写**：生成消息前，必须先运行并阅读 `git status --short`、`git diff --cached --stat`、`git diff --cached`。如果暂存区为空，不得猜测内容来编造消息。
2. **查看近期风格**：运行 `git log --oneline -10`，如果仓库已有稳定的自然写法，优先贴近；不要用外部模板覆盖仓库既有风格。
3. **只基于暂存区**：默认只分析 `--cached` 的内容。未被暂存的文件不得进入提交消息。
4. **一个提交一个逻辑变更**：如果一条消息无法用一行概括，说明暂存区混入了多个不相关的改动。此时应提示用户拆分，而不是用 `and` 或分号把多个变更塞进一个 subject。
5. **不自行执行 `git commit`**：Agent 只生成并展示消息，由用户确认后提交。如果用户明确授权 Agent 提交，使用 HEREDOC 形式传递多行消息，避免 shell 转义问题。
6. **不自创格式**：如果仓库的 `AGENTS.md` 或 `CONTRIBUTING.md` 规定了与本节不同的格式（例如纯祈使句、`[tag]` 前缀、强制 scope），以仓库规则为准；但不得添加本节未要求的额外标记。

### AI 参与披露（按仓库策略选择）

如果仓库要求披露 AI 参与，使用 footer trailer，**不要**使用 `Co-Authored-By:`：

Assisted-by: <工具名> (<模型标识>)

- 使用 `Assisted-by:` 而非 `Co-Authored-By:`。`Co-Authored-By:` 会使 AI 工具成为共同作者，并可能触发 CLA / DCO 签名问题。
- `Signed-off-by:` 只能填写人类作者，Agent 不得代填。
- 如果仓库不要求披露，省略该 trailer；不要主动添加“Generated with”类标记。
