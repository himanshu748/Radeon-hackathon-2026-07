# VHS tape for the AMD AI DevMaster demo video.
#
#   vhs scripts/demo.tape
#
# Every command below really executes in a real shell and the output is
# whatever the machine actually returns. Nothing is staged or replayed from a
# transcript. Requires these to be exported first:
#
#   VULCAN_BASE_URL        the live Radeon vLLM instance  (generation)
#   VULCAN_API_KEY         that instance's per-session key
#   VULCAN_EMBED_BASE_URL  http://localhost:11434/v1      (embeddings)
#   VULCAN_EMBED_MODEL     mxbai-embed-large

Output demo-raw.mp4

Set Shell "bash"
Set FontSize 15
Set Width 1500
Set Height 860
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 25ms

# --- 1. What is serving this ------------------------------------------
Type "# Vulcan: a codebase agent that never calls a third-party LLM API."
Enter
Sleep 2s
Type "curl -s $VULCAN_BASE_URL/models -H \"Authorization: Bearer $VULCAN_API_KEY\" | python3 -c 'import sys,json; d=json.load(sys.stdin)[\"data\"][0]; print(\"served model:\", d[\"id\"], \"| ctx:\", d[\"max_model_len\"])'"
Enter
Sleep 6s

Type "# That is vLLM on ROCm, on an AMD Radeon GPU. Generation runs there."
Enter
Sleep 2s
Type "# Embeddings run on a local model, because task=generate serves no /v1/embeddings."
Enter
Sleep 3s

# --- 2. Index a real repository ---------------------------------------
Type "vulcan index ~/demo-repo/src --name demo"
Enter
Sleep 40s

# --- 3. The agent reasons, on the GPU ---------------------------------
Type "# Now a question that needs searching, reading and citing."
Enter
Sleep 2s
Type "cd ~/demo-repo && vulcan ask 'where is the request context pushed and popped?' --name demo"
Enter
Sleep 60s

# --- 4. Measured on the GPU -------------------------------------------
Type "cd ~/n/vulcan"
Enter
Sleep 800ms
Type "# For an agent, single-stream speed is the wrong metric."
Enter
Sleep 2s
Type "# What matters is many ReAct steps running at once."
Enter
Sleep 2s
Type "vulcan bench-concurrency --label demo-live --levels 1,8"
Enter
Sleep 135s

# --- 5. Against the laptop --------------------------------------------
Type "# Same harness, run earlier on an M4 Air with Ollama:"
Enter
Sleep 2s
Type "vulcan bench-compare bench-results/ollama-m4air-concurrency.json bench-results/radeon-vllm-concurrency.json"
Enter
Sleep 14s

Type "# Laptop goes backwards on the second request. Radeon scales 7.63x to eight."
Enter
Sleep 5s
