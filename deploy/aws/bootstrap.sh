#!/usr/bin/env bash
# Keep CloudFormation UserData in sync with this script.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: sudo bash $0 <domain> <region> <log-group>" >&2
  exit 2
fi
if [[ $EUID -ne 0 ]]; then
  echo "Run this script as root (sudo)." >&2
  exit 2
fi
domain=$1
region=$2
log_group=$3
if [[ ! $domain =~ ^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$ ]] || [[ $domain != *.* ]]; then
  echo "domain must be a DNS hostname without a scheme or path" >&2
  exit 2
fi
if [[ ! $region =~ ^[a-z]{2}(-[a-z]+)+-[0-9]+$ ]] || [[ ! $log_group =~ ^[A-Za-z0-9/_.#-]+$ ]]; then
  echo "Invalid AWS region or CloudWatch log group" >&2
  exit 2
fi
. /etc/os-release
if [[ $ID != ubuntu ]] || [[ $(dpkg --print-architecture) != amd64 ]]; then
  echo "This setup requires Ubuntu amd64." >&2
  exit 2
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl gnupg unzip python3

# Docker's official repository includes the Compose v2 plugin.
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
printf 'deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n' "$VERSION_CODENAME" > /etc/apt/sources.list.d/docker.list

# Caddy's official stable repository supplies its systemd service.
curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt -o /etc/apt/sources.list.d/caddy-stable.list
chmod a+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg /etc/apt/sources.list.d/caddy-stable.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin caddy

# AWS CLI v2; skip installation on subsequent runs.
if ! command -v aws >/dev/null 2>&1 || [[ $(aws --version 2>&1) != aws-cli/2.* ]]; then
  setup_tmp=$(mktemp -d)
  trap 'rm -rf "$setup_tmp"' EXIT
  curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o "$setup_tmp/awscliv2.zip"
  unzip -q "$setup_tmp/awscliv2.zip" -d "$setup_tmp"
  "$setup_tmp/aws/install" --update
fi

# Ubuntu EC2 images normally include SSM Agent; handle an existing host too.
if systemctl list-unit-files amazon-ssm-agent.service --no-legend | grep -q '^amazon-ssm-agent'; then
  systemctl enable --now amazon-ssm-agent
else
  if ! command -v snap >/dev/null 2>&1; then apt-get install -y snapd; fi
  if ! snap list amazon-ssm-agent >/dev/null 2>&1; then
    snap install amazon-ssm-agent --classic
  fi
  snap start --enable amazon-ssm-agent
fi

install -d -m 0750 /opt/agentdesk
install -d -m 0755 /etc/docker
# Preserve unrelated daemon settings, and restart only when configuration changes.
docker_config_changed=$(python3 - "$region" "$log_group" <<'PY'
import json
import sys
from pathlib import Path

path = Path('/etc/docker/daemon.json')
config = json.loads(path.read_text()) if path.exists() else {}
previous = json.loads(json.dumps(config))
config['log-driver'] = 'awslogs'
config['log-opts'] = {
    'awslogs-region': sys.argv[1],
    'awslogs-group': sys.argv[2],
    'awslogs-create-group': 'false',
    'tag': '{{.Name}}/{{.ID}}',
}
if config != previous:
    path.write_text(json.dumps(config, indent=2) + '\n')
print('true' if config != previous else 'false')
PY
)
if [[ $docker_config_changed == true ]]; then
  dockerd --validate --config-file=/etc/docker/daemon.json
  systemctl restart docker
fi
systemctl enable --now docker

printf '%s {\n    reverse_proxy localhost:8003\n}\n' "$domain" > /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
systemctl enable --now caddy
systemctl reload caddy
echo "Host setup complete. Copy docker-compose.prod.yml and .env to /opt/agentdesk before deploying."
