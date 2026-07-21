# Claude Switcher

A small Windows utility for switching the active [Claude Code](https://claude.com/claude-code) provider (API gateway / backend) with a single menu selection.

Each provider is described by a plain `.env` file. Switching cleanly **removes every variable used by any provider** from your session and the persisted Windows user environment before loading the selected one — so no setting from a previous provider can leak into the new one. It also rewrites the `claudeCode.environmentVariables` block in `~/.claude/settings.json` to match.

## How it works

```
switch-provider.bat        ← interactive menu (run this)
        │
        ▼
switch-provider.ps1        ← does the real work
        │
        ▼
providers\*.env            ← one file per provider
        │
        ▼
%USERPROFILE%\.claude\settings.json   ← kept in sync
```

1. **Discover** — scans `providers\*.env` and collects *every* variable name used across all of them.
2. **Clear** — removes all of those variables from the current process **and** the persisted Windows *user* environment (a real delete, not a blank value).
3. **Load** — applies the selected provider's variables to the process and the user environment.
4. **Sync `settings.json`** — updates the `claudeCode.environmentVariables` array so the VS Code / Claude Code extension picks up the same values.
5. **Hand back to cmd** — the freshly loaded variables are imported into the calling `cmd` session so they take effect immediately.

## Requirements

- Windows with PowerShell (5.1 or later)
- Claude Code installed (optional — the `settings.json` step is skipped with a warning if the file is absent)

## VS Code setup

For VS Code to pick up the environment variables that the switcher writes to `~/.claude/settings.json`, open **Preferences: Open User Settings (JSON)** (Ctrl+Shift+P) and add:

```json
"chat.hookFilesLocations": {
    "~/.claude/settings.json": true
}
```

## Usage

Double-click **`switch-provider.bat`**, or from a terminal:

```bat
switch-provider.bat
```

You'll see a numbered menu built from the files in `providers\`:

```
===============================
      Claude Switcher
===============================

1. BMF-1
2. BMF-2
3. litellm
4. premiumllm

Choose Provider (1-4):
```

Pick a number and the switch is applied. Secret-looking values (containing `TOKEN`, `KEY`, `SECRET`, `AUTH`, `PASSWORD`) are masked in the output, e.g. `****XwHg`.

### Running the PowerShell script directly

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\switch-provider.ps1 -Provider BMF-1
```

| Parameter          | Required | Description                                                                                 |
| ------------------ | -------- | ------------------------------------------------------------------------------------------- |
| `-Provider`        | Yes      | Provider name — matches `providers\<name>.env` (without the extension).                      |
| `-SessionVarsFile` | No       | Path to a file that receives `NAME=VALUE` lines so a calling `cmd` session can re-import them. |

## Adding a provider

Create a new file in `providers\`, e.g. `providers\myprovider.env`:

```env
# Comments (lines starting with #) and blank lines are ignored.
ANTHROPIC_BASE_URL=https://my-gateway.example.com
ANTHROPIC_MODEL=claude-haiku-4-5
ANTHROPIC_AUTH_TOKEN=sk-...
```

It appears in the menu automatically the next time you run the switcher — no other changes needed.

**`.env` format**
- One `KEY=VALUE` per line.
- Everything after the first `=` is the value (so values may contain `=` and `:`, e.g. custom headers).
- Lines beginning with `#` and blank lines are skipped.

## Included providers

| File              | Backend                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| `BMF-1.env`       | Vertex-style endpoint (`CLAUDE_CODE_USE_VERTEX=1`) at `aoai-farm.bosch-temp.com`. Primary auth token.       |
| `BMF-2.env`       | Same as `BMF-1.env` but using the secondary auth token.                 |
| `litellm.env`     | LiteLLM gateway (`claude-haiku-4-5`), `x-bmf-sticky-session-instance: 01` header. |
| `premiumllm.env`  | LiteLLM gateway, separate auth token.                                   |

## Security note

⚠️ The `providers\*.env` files contain **live API tokens in plain text**. Keep this folder private:

- Do **not** commit real tokens to version control. Add `providers/*.env` to `.gitignore` and commit `*.env.example` templates instead.
- Rotate any token that may have been exposed.

## Troubleshooting

| Symptom                                             | Cause / fix                                                                 |
| --------------------------------------------------- | -------------------------------------------------------------------------- |
| `No provider .env files found`                      | The `providers\` folder is empty or missing.                               |
| `ERROR: <name>.env not found!`                      | The provider name doesn't match a file in `providers\`.                     |
| `WARNING: ... settings.json not found`              | Claude Code isn't installed, or `~/.claude/settings.json` doesn't exist.    |
| New variables don't appear in a **separate** shell  | Other shells read the environment at launch — open a new terminal.          |
