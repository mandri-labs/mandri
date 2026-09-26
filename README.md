# Mandri

Mandri is a local control plane for AI agents. It connects a desktop client and a
command-line interface to a local daemon that manages sessions, providers and
agent runtimes.

This is beta software. Interfaces may change between beta releases.

## Installation

Install the Python package into an isolated tool environment with Python 3.13:

```sh
uv tool install --python 3.13 "mandri==0.1.0b0"
```

The package provides two commands:

```sh
mandri --help
mandri-daemon --help
mandri-daemon serve
```

The daemon stores configuration and its database in `~/.mandri`. Use
`--base-dir` to select a different data directory. Its default address is
`http://127.0.0.1:8787`.

## Provider setup

Register a provider and enter its API key at the interactive prompt:

```sh
mandri provider add openrouter
mandri provider list
mandri provider models openrouter
```

Credentials are verified when added. Use `--key "$OPENROUTER_API_KEY"` to
supply a key from an environment variable, or `--no-verify` to skip the initial
verification. To verify again later, run `mandri provider verify openrouter`.

The provider kind is inferred when its name matches a supported kind:
`openrouter`, `opencode`, `opencode_go`, `openai`, `anthropic`, `gemini`,
`ollama`, `lm_studio` or `custom`. For a different name, pass `--kind`:

```sh
mandri provider add my-provider --kind openrouter
```

For `ollama`, `lm_studio` and `custom`, also supply the API base URL with
`--base`. All provider commands accept `--base-dir` to select a data directory
other than `~/.mandri`; use the same directory when launching a harness.
Remove unused credentials with `mandri provider remove <name>`.

## Run a harness

Choose a model ID from `mandri provider models <name>`, then launch a harness:

```sh
mandri run --model "openrouter/<model-id>" codex
```

Replace `<model-id>` with the exact model ID returned by the provider, including
any slashes. The full reference is `<provider-name>/<model-id>`; for a provider
registered as `my-provider`, use `my-provider/<model-id>`.

Supported harnesses are `codex`, `claude`, `opencode`, `agy` and `pi`. Codex uses the
runtime bundled with Mandri. Claude Code and OpenCode must be installed and
available on `PATH`; `agy` requires the Antigravity CLI and `pi` requires the Pi CLI.

`mandri run` starts the local daemon if needed and runs the harness in the
foreground through the Mandri gateway. It uses the current working directory
unless `--cwd` is set. Put Mandri options before the harness name, and pass
harness-specific arguments after `--`:

```sh
mandri run --model "openrouter/<model-id>" --cwd /path/to/project codex
mandri run --model "openrouter/<model-id>" --effort high claude
mandri run --model "openrouter/<model-id>" codex -- --help
```

Reasoning effort support depends on the selected model. `mandri run` requires
host execution with privacy mode set to `none`; if the configured protection
requires managed execution, use `mandri sessions start` instead.

## License

Mandri is licensed under Apache-2.0. See `LICENSE` for details.
