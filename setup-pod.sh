#!/bin/bash
# setup-pod.sh — Run inside a RunPod pod to set up autoinfer
#
# Usage: curl the repo and pipe, or just paste into the pod terminal:
#   bash setup-pod.sh
set -euo pipefail

echo "=== Installing Node.js ==="
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs

echo "=== Installing Claude Code ==="
npm install -g @anthropic-ai/claude-code

echo "=== Installing GitHub CLI ==="
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
apt update && apt install gh -y

echo "=== Authenticating with GitHub ==="
gh auth login

echo "=== Cloning autoinfer ==="
cd /workspace
gh repo clone pavanyellow/autoinfer
cd autoinfer

echo "=== Running prepare.py (downloads model + dataset) ==="
python prepare.py

echo "=== Creating non-root user (Claude Code refuses to run as root) ==="
useradd -m -s /bin/bash coder
cp -r /root/.anthropic /home/coder/.anthropic 2>/dev/null || true
chown -R coder:coder /home/coder /workspace

echo ""
echo "=== Setup complete! ==="
echo "Run Claude Code with:"
echo "  su - coder -c \"cd /workspace/autoinfer && claude --dangerously-skip-permissions\""
