# Codex

OpenAI Codex CLI agent deployed via NPM in sandbox environments.

## Architecture

```
FRAMEWORK                                 SANDBOX
lifecycle.py                              cua-computer-server
  install()   -> npm install, write config.toml, verify codex --version
  launch()    -> codex exec --json (NDJSON pipe, stdin from prompt file)
                   +-- CUA MCP Server (stdio)
                        +-- CUA HTTP API (desktop/shell)
  parse_artifacts() <- transcript.jsonl, stderr.log
```

Codex runs as a headless CLI process with the prompt piped via stdin. The agent
reaches the sandbox desktop/filesystem through the CUA MCP Server configured as
a stdio transport in `~/.codex/config.toml`.

## Source Strategy

The default is the official `@openai/codex@0.163.0` package. The installer
checks `codex --version`, installs the configured version if necessary, and
verifies it again. The legacy fork remains selectable for existing experiments
with `fork_version` and its platform-specific binary URLs.

## Providers

| Provider | Auth | Model ID |
|---|---|---|
| `direct` | `OPENAI_API_KEY` | `gpt-5.4` |
| `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-5.4` |

Routing is selected by `provider`. Configuration is written to
`~/.codex/config.toml`:

```toml
model_provider = "openrouter"

[model_providers.openrouter]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
env_key = "OPENROUTER_API_KEY"
```

## Bridge

CUA MCP Server, stdio transport. Written to `~/.codex/config.toml`:

```toml
[mcp_servers.cua]
type = "stdio"
command = "/usr/local/bin/node"
args = ["/home/user/cua_mcp_server/src/index.js"]
```

## Config

Config fields (standalone dataclass):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `sandbox_mode` | str | `"danger-full-access"` | Codex sandbox policy |
| `yolo` | bool | `true` | Bypass all approval prompts |
| `reasoning_effort` | str | `"high"` | Model reasoning effort hint |
| `cli_version` | str | `"0.163.0"` | Pinned official npm package version |
| `fork_version` | str | `""` | Optional legacy fork version to reproduce an existing run |
| `patched_binary_url` / `patched_binary_url_windows` | str | legacy release URLs | Fork binaries, used only when `fork_version` is set |
| `capture_session` | bool | `true` | Preserve the matching Codex session JSONL |
| `rate_limit_resume_retries` | int | `4` | Resume the same session after HTTP 429 |
| `rate_limit_backoff_base_s` / `rate_limit_backoff_max_s` | float | `30` / `240` | Resume backoff bounds in seconds |

## Logs

Codex `--json` outputs NDJSON with event types:
- `thread.started` -- session ID
- `turn.started` / `turn.completed` -- turn boundaries with usage stats
- `item.started` / `item.completed` -- tool calls, messages, reasoning
- `error` -- critical failures

The Codex CLI's `--json` stream does not expose opaque Responses reasoning
items. When `capture_session` is enabled (the default), the deployer copies the
matching `~/.codex/sessions/<date>/*.jsonl` file into
`origin_log/codex/codex_session/` after the child exits. This preserves
`encrypted_content` for replay/audit without attempting to decrypt or project
it into `trajectory.json`.

Item types: `agent_message`, `reasoning`, `command_execution`,
`mcp_tool_call`, `file_change`, `web_search`, `error`.

## Execution

- CLI: `codex exec --model <model> --json --dangerously-bypass-approvals-and-sandbox`
- Process: detached via `setsid` (Linux) / `Start-Process` (Windows)
- Polling: every 2s via Popen.poll()
- Rate limiting: after Codex exhausts its internal retries, resume the same
  session in the same VM up to four times after 30, 60, 120, and 240 seconds
- Timeout: terminates process, then kills after grace period
- Exit code: nonzero -> `status: "failed"`

## Legacy Fork Binary Paths

npm 11.x stopped hoisting platform deps so the nested copy is the one
`codex.js`'s `require.resolve` actually picks. Both paths are checked
and whichever exists gets replaced:

- Linux top-level: `/usr/local/lib/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex/codex`
- Linux nested: `/usr/local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex/codex`
- Windows top-level: `...\npm\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\codex\codex.exe`
- Windows nested: `...\npm\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\codex\codex.exe`

## Smoke Test

```bash
uv run python -m ale_run run experiments/smoke_codex_docker.yaml
```
