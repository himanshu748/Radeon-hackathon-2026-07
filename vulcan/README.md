# Vulcan

**Track 2, Private AI Agents. AMD AI DevMaster Hackathon 2026.**

A developer-productivity agent for codebases that cannot leave the machine.
It indexes a repository, then reasons over it with a ReAct tool loop:
semantic search, file reads, grep, test runs and edits, answering with
`file:line` citations. Agent reasoning and generation run on an AMD Radeon GPU
through vLLM on ROCm. Nothing is sent to a cloud API.

To be precise about the split, because it is easy to overstate: `vllm serve
Qwen/Qwen3-8B` runs with task=generate and exposes no `/v1/embeddings` route
(verified, it returns 404). Embeddings therefore come from a separate
endpoint, and since Radeon Cloud allows one active instance per account, the
measured setup runs generation on the GPU and embeddings on a local model.
Both endpoints are operator-controlled so no source leaves your machines, but
only generation is GPU-served here. See `spec-document.md` section 2.1.

Source: **https://github.com/himanshu748/vulcan**

## Why this exists

Developers on private or regulated codebases have two bad options: paste
source into a hosted LLM, or go without assistance. Vulcan removes the
choice. The agent is backend-agnostic over one OpenAI-compatible endpoint, so
the same binary runs against Ollama on a laptop during development and vLLM
on ROCm in production. Switching is one environment variable, which is also
what made the measurements below a controlled swap rather than a rewrite.

## Quick start

```bash
pip install -e .
cp .env.example .env          # then edit
vulcan index ~/code/myproject
cd ~/code/myproject
vulcan ask "where is auth token validation done?"
```

Against the Radeon box:

```bash
export VULCAN_BASE_URL=https://<host>/spaces/<instance-id>/8000/v1
export VULCAN_API_KEY=<per-instance key>
export VULCAN_MODEL=Qwen/Qwen3-8B
```

## Architecture

```
user -- CLI -- Agent (ReAct, JSON tool protocol)
                 |- RAG index (SQLite + cosine, no vector DB dependency)
                 |- Tools (search_code, read_file, grep, run_cmd*, write_file)
                 |- Memory (durable per-project notes)
                 `- LLM client -- OpenAI-compatible endpoint
                                    `- vLLM on ROCm / Radeon GPU
```

`*` allowlisted commands only; file access is path-jailed to the indexed root.

The JSON tool protocol matters: it means the agent works on backends without
native tool-calling support, which is most local serving stacks.

## ROCm work

Full detail in [`spec-document.md`](spec-document.md), raw JSON in
[`bench-results/`](bench-results/), all of it reproducible with the harness
that ships in the repo:

```bash
vulcan bench              --label <name>    # decode: TTFT + generation rate
vulcan bench-concurrency  --label <name>    # aggregate throughput vs parallel load
vulcan bench-prefill      --label <name>    # TTFT vs input length
vulcan bench-compare      <a.json> <b.json> # markdown table
```

Two findings are worth the reviewer's time:

1. **Qwen3 thinking mode is the wrong default for an agent.** Disabling it
   per request via `chat_template_kwargs` cut wall-clock latency 3.5x on the
   same hardware with *identical* generation rate. The win is token count,
   not speed.
2. **Concurrency is where the GPU argument actually lives.** The laptop's
   aggregate throughput *falls* when a second request arrives and TTFT grows
   nearly 10x, because Ollama does not batch. vLLM's continuous batching on
   ROCm keeps the curve going the right way.

## Honest measurement

Stated plainly because it affects how the numbers read:

- Radeon figures are collected through the Radeon Cloud HTTPS proxy. The
  instance SSH port is refused from the client network and the Jupyter port
  returns 403, so no on-box run was possible. Proxy round-trip was measured
  separately and is included in every reported TTFT.
- "chunks/s" counts streamed SSE content deltas, not tokenizer tokens.
- The Radeon runs Qwen3-8B and the laptop runs qwen3:4b-instruct. Absolute
  decode numbers are therefore not a clean hardware isolation and are not
  presented as one. See `spec-document.md` section 4.2.

## Dependencies

Python 3.10 or newer. Four runtime packages, declared in `pyproject.toml`:

| package | version | used for |
|---|---|---|
| `httpx` | >=0.27 | streaming HTTP to the OpenAI-compatible endpoint |
| `numpy` | >=1.26 | cosine similarity over the embedding index |
| `typer` | >=0.12 | CLI |
| `rich` | >=13.7 | terminal output |

Development adds `pytest` >=8.0. There is no vector database, no LangChain and
no agent framework: the index is SQLite plus numpy, and the ReAct loop is about
a hundred lines in `vulcan/agent.py`.

A backend is also required, one of:

- **Ollama** for local development, serving both chat and embeddings
- **vLLM on ROCm** for GPU generation, plus a local embeddings endpoint

## Environment configuration

Every value is read from the environment; nothing is hardcoded. Copy
`.env.example` to `.env` and fill it in.

| variable | default | meaning |
|---|---|---|
| `VULCAN_BASE_URL` | `http://localhost:11434/v1` | chat endpoint |
| `VULCAN_API_KEY` | `local` | key for that endpoint |
| `VULCAN_MODEL` | `qwen3:4b-instruct` | generation model |
| `VULCAN_EMBED_BASE_URL` | falls back to `VULCAN_BASE_URL` | embeddings endpoint |
| `VULCAN_EMBED_API_KEY` | falls back to `VULCAN_API_KEY` | key for that endpoint |
| `VULCAN_EMBED_MODEL` | `mxbai-embed-large` | embedding model |
| `VULCAN_ENABLE_THINKING` | unset (sends nothing) | `false` cuts agent-step latency ~3.5x on Qwen3 via vLLM |
| `VULCAN_TEMPERATURE` | `0.2` | sampling temperature |
| `VULCAN_MAX_STEPS` | `12` | ReAct step ceiling |
| `VULCAN_DATA_DIR` | `~/.vulcan` | index and memory storage |

## Tests

```bash
pip install -e ".[dev]"
pytest
```
