# Easy deploy: GitHub change -> MAADS VPS deploy

Goal: a tiny, explicit GitHub change triggers deployment of `main` to the VPS serving `https://maads.mirogeorgiev.eu`.

## Trigger

Edit this file on GitHub:

- `.deploy/maads-webapp.trigger`

Change `last_requested_utc`, commit directly to `main`, and GitHub Actions runs `.github/workflows/easy-deploy.yml`.

The workflow SSHes into the VPS and runs:

- `/root/work/deploy/maads-webapp/deploy.sh`

## Required GitHub repository secrets

Set these under GitHub repo -> Settings -> Secrets and variables -> Actions -> Repository secrets:

- `VPS_SSH_HOST`: VPS hostname or IP
- `VPS_SSH_USER`: SSH user; normally `root` for this VPS profile
- `VPS_SSH_PRIVATE_KEY`: private key allowed to SSH to the VPS
- `VPS_SSH_PORT`: optional; defaults to `22`

Use a dedicated deploy key if possible. Do not paste these values into commits, issues, or logs.

## What the VPS deploy script does

1. Takes a deploy lock so two deploys cannot overlap.
2. Verifies prerequisites and confirms the repo remote is `git@github.com:migush/agentic-crisp-dm.git`.
3. Fetches `origin/main` and verifies the requested SHA is on `main`.
4. Backs up tracked local diff/status under `/root/work/agentic-crisp-dm/backups/deploy/`.
5. Resets the working tree to the requested commit.
6. Installs backend dependencies into `.venv`.
7. Builds the React frontend from `webapp/frontend`.
8. Publishes frontend files to `/srv/maads-webapp/releases/<timestamp>-<sha>` and updates `/srv/maads-webapp/dist`.
9. Restarts `maads-webapp.service` and `maads-dashboard.service`.
10. Verifies local API/dashboard and public HTTPS homepage.

## Manual deploy from the VPS

```bash
DEPLOY_BRANCH=main /root/work/deploy/maads-webapp/deploy.sh
```

Deploy a specific commit that is already on `main`:

```bash
DEPLOY_BRANCH=main DEPLOY_SHA=<commit-sha> /root/work/deploy/maads-webapp/deploy.sh
```

Preflight only:

```bash
CHECK_ONLY=1 /root/work/deploy/maads-webapp/deploy.sh
```

## Rollback

Frontend rollback to the previous published frontend symlink, if present:

```bash
ln -sfn "$(readlink -f /srv/maads-webapp/releases/previous)" /srv/maads-webapp/dist
systemctl restart maads-webapp.service maads-dashboard.service
```

Code rollback to a known good commit on `main`:

```bash
DEPLOY_BRANCH=main DEPLOY_SHA=<known-good-sha> /root/work/deploy/maads-webapp/deploy.sh
```

If a deploy overwrote local tracked edits, inspect backups in:

- `/root/work/agentic-crisp-dm/backups/deploy/`

## Notes

- The trigger is intentionally a specific file, not every push to `main`, so normal development merges do not deploy unless the trigger file changes.
- The workflow does not expose any new public ports; it uses the existing Caddy and systemd deployment.
- Untracked local files are not deleted by the deploy script. Tracked local modifications are backed up before reset.
