# Running LineGuard's AI on AMD

LineGuard's language model runs on AMD Instinct GPUs. The app reaches it through any
OpenAI-compatible endpoint, so you can use either of two options. Only the four `LLM_*`
variables change between them.

## Option A: AMD Token Factory (the hackathon API, no credit card needed)

1. Sign in at https://notebooks.amd.com/tokenfactory with your AMD AI Developer Program account.
2. Create an API key, then copy the **base URL** (it ends in `/v1`) and a **model name** from the model list.
3. Put them in `.env`:

   ```bash
   LLM_BASE_URL=https://<base-url-from-token-factory>/v1
   LLM_API_KEY=<key>
   LLM_MODEL=<model name>
   LLM_HARDWARE=AMD Instinct GPU · AMD Token Factory
   ```

4. Check the connection and record numbers for the pitch:

   ```bash
   set -a; source .env; set +a
   python -m scripts.benchmark_llm --n 6
   ```

## Option B: your own vLLM server on an AMD Instinct MI300X (AMD Developer Cloud)

1. At https://devcloud.amd.com, create a GPU droplet (1× MI300X) from the **vLLM** quick-start image.
2. SSH in, then serve an open model. Any vLLM-supported instruct model works. Qwen3-30B-A3B fits easily in the
   MI300X's 192 GB and is fast because it is a mixture-of-experts model:

   ```bash
   export KEY=$(openssl rand -hex 16); echo $KEY
   docker run -d --name vllm --network=host --ipc=host --shm-size 16G \
     --device=/dev/kfd --device=/dev/dri --group-add video \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     rocm/vllm:latest \
     vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --host 0.0.0.0 --port 8000 \
       --api-key $KEY --max-model-len 16384
   docker logs -f vllm      # wait for "Application startup complete"
   ```

   (If the droplet image already has vLLM installed, you can run the same `vllm serve ...` command directly.
   Check `rocm-smi` to confirm the GPU is in use.)

3. Open port 8000 in the droplet firewall, or tunnel it with `ssh -L 8000:localhost:8000 root@<ip>`.
4. Configure the app:

   ```bash
   LLM_BASE_URL=http://<droplet-ip>:8000/v1
   LLM_API_KEY=<KEY>
   LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
   LLM_HARDWARE=AMD Instinct MI300X · ROCm · vLLM
   ```

Destroy the droplet when you finish, because it is billed by the hour.

## Running the whole app inside AMD AI Notebooks

To keep everything on AMD infrastructure, open a terminal in JupyterLab at notebooks.amd.com, then run:

```bash
git clone <your-repo-url> lineguard && cd lineguard
pip install -r requirements.txt
cp .env.example .env   # fill in the Token Factory values
set -a; source .env; set +a
uvicorn lineguard.api:app --host 0.0.0.0 --port 8000
```

Then open the notebook server's proxy URL for port 8000 (usually `.../proxy/8000/`).
