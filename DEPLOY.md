# Deployment

PRs run Python 3.12 tests (including read-scoped Postgres tests), a cached Docker build, and a 120-scenario eval gate when `OPENROUTER_API_KEY` is available. Eval metrics, baseline deltas, failure traces, and triage appear in the job summary, one sticky PR comment, and the results artifact. Fork PRs without secrets skip eval cleanly.

Pushes to `main` run tests, publish SHA and `latest` image tags to ECR, then deploy to EC2 through SSM and check HTTPS `/health`. Deployments run one at a time. Caddy terminates TLS and proxies to `127.0.0.1:8003`; Compose runs the app, pgvector Postgres, and internal Redis with named data volumes. Container logs go to CloudWatch `/agentdesk` with 30-day retention; optional Langfuse Cloud keys enable eval and application traces. The stack opens only ports 80/443 and requires IMDSv2; administration uses SSM without SSH.

## One-time setup

1. With AWS CLI credentials authorized to create the stack, choose a region and deploy:

   ```bash
   export AWS_REGION=us-east-1
   aws cloudformation deploy \
     --region "$AWS_REGION" \
     --stack-name agentdesk \
     --template-file deploy/aws/cloudformation.yml \
     --capabilities CAPABILITY_NAMED_IAM \
     --parameter-overrides GitHubRepo=OWNER/REPO DomainName=support.example.com
   aws cloudformation describe-stacks --region "$AWS_REGION" \
     --stack-name agentdesk --query 'Stacks[0].Outputs' --output table
   ```

   Add `CreateOidcProvider=false` to the parameter overrides if the account already has the GitHub OIDC provider (its audience must include `sts.amazonaws.com`). `InstanceType` defaults to `t3.small`; overrides must support x86_64. The template creates its own VPC/public subnet and an Elastic IP. ECR images and the log group are retained when deleting the stack; the instance's data disk is deleted, so back it up first.

2. Point the domain's DNS A record to output `PublicIp`. Wait for DNS propagation and host initialization. Caddy obtains and renews its TLS certificate automatically.

3. Install the local AWS Session Manager plugin and open a session using an operator identity with SSM session permissions:

   ```bash
   aws ssm start-session --region "$AWS_REGION" --target INSTANCE_ID
   ```

   In the session, run `sudo cloud-init status --wait`, then `sudo -i` and `cd /opt/agentdesk`. Copy the contents of this revision's `docker-compose.prod.yml` into that directory through the session (`cat > docker-compose.prod.yml`, paste, Ctrl-D). Create `.env` the same way from `.env.example`, using `umask 077` before writing, then `chmod 600 .env`.

   Set `OPENROUTER_API_KEY`, unique `AUTH_SECRET`, `SERVICE_TOKEN`, `SUPERVISORS`, and `DEMO_CUSTOMER_PASSWORD`. Add strong `POSTGRES_PASSWORD` and `READER_DB_PASSWORD` values (for example, hex strings from `openssl rand -hex 32`; passwords are interpolated into database URLs). Add Langfuse Cloud keys and `LANGFUSE_HOST` if using traces. GitHub secrets are for CI; they are not copied into the server's `.env`. Keep database passwords stable across deploys; changing the Postgres env value alone does not rotate an existing database password.

   For an existing Ubuntu amd64 host, attach the instance policies shown in the template, ensure SSM/outbound internet access, create the log group, and run `sudo bash deploy/aws/bootstrap.sh support.example.com us-east-1 /agentdesk`. The bootstrap is idempotent; it installs Docker/Compose, AWS CLI v2, SSM Agent, Caddy, and the default Docker CloudWatch log driver. Reconfiguring Docker logging restarts Docker; recreate existing containers to adopt the new driver.

4. Configure GitHub repository Actions settings:

   | Kind | Name | Value |
   | --- | --- | --- |
   | Secret | `OPENROUTER_API_KEY` | Eval API key |
   | Secret | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | Optional Langfuse Cloud credentials |
   | Secret | `LANGFUSE_HOST` | Optional; defaults to `https://cloud.langfuse.com` |
   | Secret | `AWS_DEPLOY_ROLE_ARN` | Stack output `DeployRoleArn` |
   | Variable | `AWS_REGION` | Stack region |
   | Variable | `EC2_INSTANCE_ID` | Stack output `InstanceId` |
   | Variable | `APP_URL` | Domain only, e.g. `support.example.com` |
   | Variable | `ECR_REPOSITORY` | Repository name/path from `EcrRepositoryUri`, **without** the registry hostname |

   The deploy role trusts only `repo:OWNER/REPO:ref:refs/heads/main`. Run manual deployments from `main`. After the host files are ready, push to `main` or run the **deploy** workflow on `main` with an empty `image_tag` to build and deploy. SSM output and the HTTPS smoke check determine deployment success. This guide and template do not indicate that a deployment has been performed.

## Eval baseline and thresholds

Store a reviewed run as the baseline, then include it in a subsequent PR:

```bash
cp eval/results/REVIEWED_TAG.json eval/baseline.json
```

The gate displays deltas when that file exists. Defaults are `MIN_TASK_SUCCESS=0.80`, `MIN_TOOL_ACCURACY=0.85`, and `MAX_P95_S=5.0`. Override them as environment variables on the CI eval job or gate step. Task success and tool-call accuracy must meet their minimum; p95 turn latency must not exceed its maximum. Missing or invalid gate metrics fail. Faithfulness, tokens, cost, and group/category results are reported for review.

## Rollback

Run **deploy → Run workflow** on `main`, with `image_tag` set to a previously deployed commit SHA that still exists in ECR. Tests run first; image building is skipped. SSM pulls that tag, starts Compose, and the same health check runs. ECR retains the newest 20 images, so older tags may have expired. Image rollback does not restore database state or reverse schema changes; a failed smoke check requires operator review or a manual rollback.

## Backups and operations

In an SSM session, become root and dump Postgres before upgrades and on your backup schedule:

```bash
cd /opt/agentdesk
export IMAGE="$(cat .deployed-image)"
install -d -m 700 backups
umask 077
docker compose -f docker-compose.prod.yml exec -T db \
  pg_dump -U agentdesk -d agentdesk -Fc > "backups/agentdesk-$(date -u +%Y%m%dT%H%M%SZ).dump"
```

Copy backups off the instance to protected storage with an operator identity and verify restoration with `pg_restore` against a separate database. Postgres contains business data and LangGraph checkpoints; Redis's named volume contains conversation memory. Preserve both volumes for full recovery. Avoid `docker compose down -v`. There is no automatic backup scheduler or failover in this single-instance stack.

For manual Compose maintenance, export `IMAGE` from `.deployed-image` as above. For Compose or environment changes, update the files through SSM before the next deployment. Check `/health`, container health, CloudWatch logs, and Langfuse traces after releases.

## Local alternative

```bash
cp .env.example .env
# Set the API key and replace the example credentials.
docker compose up -d --build
```

The local app is available at `http://localhost:8003`. Local Compose builds the image and does not require AWS or Caddy.

References: [Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/), [Caddy installation](https://caddyserver.com/docs/install), [SSM command waiter](https://docs.aws.amazon.com/cli/latest/reference/ssm/wait/command-executed.html).
