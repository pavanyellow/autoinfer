#!/bin/bash
# setup-runpod.sh — Spin up a RunPod GPU instance for autoinfer
#
# Prerequisites:
#   - runpodctl installed (https://github.com/runpod/runpodctl)
#   - RunPod API key configured: runpodctl doctor
#   - A network volume already created on RunPod
#
# Usage:
#   ./setup-runpod.sh                    # uses defaults
#   GPU=A100 ./setup-runpod.sh           # override GPU type
#   VOLUME_ID=abc123 ./setup-runpod.sh   # override volume

set -euo pipefail

# ---- Configuration (override via env vars) ----
POD_NAME="${POD_NAME:-autoinfer}"
GPU="${GPU:-A40}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-30}"
VOLUME_ID="${VOLUME_ID:-htveal2af9}"
DATACENTER="${DATACENTER:-EU-SE-1}"
TEMPLATE="${TEMPLATE:-runpod-torch-v280}"  # PyTorch 2.8.0 + CUDA 12.8.1 + Ubuntu 24.04

# ---- GPU ID mapping ----
declare -A GPU_MAP=(
  [A40]="NVIDIA A40"
  [A100]="NVIDIA A100 80GB PCIe"
  [A100-SXM]="NVIDIA A100-SXM4-80GB"
  [H100]="NVIDIA H100 80GB HBM3"
  [H100-PCIe]="NVIDIA H100 PCIe"
  [RTX4090]="NVIDIA GeForce RTX 4090"
)

GPU_ID="${GPU_MAP[$GPU]:-$GPU}"

echo "=== AutoInfer RunPod Setup ==="
echo "  Pod name:       $POD_NAME"
echo "  GPU:            $GPU_ID"
echo "  Template:       $TEMPLATE"
echo "  Container disk: ${CONTAINER_DISK_GB}GB"
echo "  Network volume: $VOLUME_ID"
echo "  Datacenter:     $DATACENTER"
echo ""

# ---- Create pod via GraphQL API ----
# runpodctl doesn't support attaching existing network volumes,
# so we use the API directly.
API_KEY=$(grep -oP 'apiKey\s*=\s*"\K[^"]+' ~/.runpod/config.toml 2>/dev/null || echo "")
if [ -z "$API_KEY" ]; then
  echo "Error: No API key found. Run 'runpodctl doctor' first."
  exit 1
fi

echo "Creating pod..."
RESPONSE=$(curl -s --request POST \
  --header 'content-type: application/json' \
  --header "authorization: Bearer $API_KEY" \
  --url 'https://api.runpod.io/graphql' \
  --data "{\"query\": \"mutation { podFindAndDeployOnDemand(input: { name: \\\"$POD_NAME\\\", gpuTypeId: \\\"$GPU_ID\\\", gpuCount: 1, cloudType: SECURE, containerDiskInGb: $CONTAINER_DISK_GB, networkVolumeId: \\\"$VOLUME_ID\\\", templateId: \\\"$TEMPLATE\\\", dataCenterId: \\\"$DATACENTER\\\" }) { id name } }\"}")

POD_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['podFindAndDeployOnDemand']['id'])" 2>/dev/null)

if [ -z "$POD_ID" ]; then
  echo "Error creating pod:"
  echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
  exit 1
fi

echo "Pod created: $POD_ID"
echo ""

# ---- Wait for SSH ----
echo "Waiting for pod to start..."
for i in $(seq 1 30); do
  SSH_INFO=$(runpodctl ssh info "$POD_ID" 2>/dev/null || echo "{}")
  IP=$(echo "$SSH_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ip',''))" 2>/dev/null)
  PORT=$(echo "$SSH_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('port',''))" 2>/dev/null)

  if [ -n "$IP" ] && [ -n "$PORT" ]; then
    if ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -o ConnectTimeout=5 "root@$IP" -p "$PORT" "echo ok" 2>/dev/null; then
      echo ""
      echo "=== Pod is ready! ==="
      echo ""
      echo "SSH command:"
      echo "  ssh -i ~/.runpod/ssh/RunPod-Key-Go root@$IP -p $PORT"
      echo ""
      echo "Once connected, run:"
      echo "  # Install tools"
      echo "  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg"
      echo "  echo 'deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main' | tee /etc/apt/sources.list.d/github-cli.list > /dev/null"
      echo "  apt update && apt install gh -y"
      echo "  npm install -g @anthropic-ai/claude-code"
      echo ""
      echo "  # Clone and run"
      echo "  cd /workspace"
      echo "  gh repo clone pavanyellow/autoinfer && cd autoinfer"
      echo "  python prepare.py"
      echo "  python bench.py --baseline --experiment baseline"
      echo "  claude 'Read program.md and follow the experiment loop. Start from the baseline and run experiments autonomously.'"
      exit 0
    fi
  fi
  printf "."
  sleep 10
done

echo ""
echo "Pod didn't become reachable in 5 minutes. Check RunPod dashboard."
echo "Pod ID: $POD_ID"
exit 1
