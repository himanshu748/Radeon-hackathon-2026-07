# VHS tape for the AMD AI DevMaster demo video.
#
#   bash scripts/record-demo.sh
#
# Every command below really executes in a real shell and the output is
# whatever the machine actually returns. Nothing is staged or replayed from a
# transcript. Credentials come from .env; see scripts/record-demo.sh.
#
# VHS drops frames while the screen is idle, so runtime is driven by streamed
# output rather than by Sleep. Adding real work is what lengthens the video.

Output demo-raw.mp4

Set Shell "bash"
Set FontSize 15
Set Width 1500
Set Height 860
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms

# --- 1. What is actually serving this ---------------------------------
Type "# Vulcan: a codebase agent that never calls a third-party LLM API."
Enter
Sleep 2s
Type "bash scripts/show-endpoint.sh"
Enter
Sleep 7s

Type "# vLLM on ROCm, on an AMD Radeon GPU. Generation runs there."
Enter
Sleep 2s
Type "# Embeddings run locally: task=generate serves no /v1/embeddings route."
Enter
Sleep 3s

# --- 2. Index a real repository ---------------------------------------
Type "vulcan index ~/demo-repo/src --name demo"
Enter
Sleep 45s

# --- 3. The agent reasons, generating on the GPU -----------------------
Type "cd ~/demo-repo"
Enter
Sleep 800ms
Type "vulcan ask 'where is the request context pushed and popped?' --name demo"
Enter
Sleep 70s

Type "vulcan ask 'how does the CLI discover the application factory?' --name demo"
Enter
Sleep 70s

Type "cd ~/n/vulcan"
Enter
Sleep 800ms

# --- 4. Decode speed on the GPU ---------------------------------------
Type "# Time to first token and generation rate, three prompt sizes."
Enter
Sleep 2s
Type "vulcan bench --label demo-decode --repeats 1"
Enter
Sleep 100s

# --- 5. The axis that actually matters --------------------------------
Type "# For an agent, single-stream speed is the wrong metric."
Enter
Sleep 2s
Type "# What matters is many ReAct steps running at once."
Enter
Sleep 2s
Type "vulcan bench-concurrency --label demo-live --levels 1,4,8"
Enter
Sleep 190s

# --- 6. Context budget -------------------------------------------------
Type "# And how much retrieved context costs before the model even replies."
Enter
Sleep 2s
Type "vulcan bench-prefill --label demo-prefill --words 512,2048,8192"
Enter
Sleep 75s

# --- 7. Against the laptop --------------------------------------------
Type "# Same harness, run earlier on an M4 Air with Ollama:"
Enter
Sleep 2s
Type "vulcan bench-compare bench-results/ollama-m4air-concurrency.json bench-results/radeon-vllm-concurrency.json"
Enter
Sleep 15s

Type "# The laptop goes backwards on the second request."
Enter
Sleep 3s
Type "# The Radeon scales 7.63x out to eight, and TTFT improves."
Enter
Sleep 5s
