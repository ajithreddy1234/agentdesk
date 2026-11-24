# Deploying

The app ships as Docker Compose, so any Linux VM with Docker works (AWS EC2, GCP, DigitalOcean, a lab server).

## Single VM (e.g. AWS EC2 t3.small, Ubuntu 24.04)

1. Open ports 22, 80 and 443 in the security group.
2. Install Docker: `curl -fsSL https://get.docker.com | sh`, then `sudo usermod -aG docker $USER` and log in again.
3. `git clone <this repo> && cd <repo> && cp .env.example .env`, then set `OPENROUTER_API_KEY` and replace every secret: `AUTH_SECRET`, `SERVICE_TOKEN`, `SUPERVISORS`, `DEMO_CUSTOMER_PASSWORD`.
4. `docker compose up -d --build`. The app listens on port 8003.
5. HTTPS with Caddy (automatic Let's Encrypt). Point a DNS name at the VM, then:
   ```bash
   sudo apt install -y caddy
   echo 'your.domain.com { reverse_proxy localhost:8003 }' | sudo tee /etc/caddy/Caddyfile
   sudo systemctl reload caddy
   ```
6. Updates: `git pull && docker compose up -d --build`. Data lives in named Docker volumes and survives rebuilds.

## Checklist

- Never commit `.env`; it is git-ignored.
- Back up the database volume (`docker compose exec db pg_dump ...`) if you keep real data.
- `GET /health` reports whether the LLM key is configured; use it for uptime checks.
