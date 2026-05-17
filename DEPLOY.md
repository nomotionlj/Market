# Deploying to Streamlit Cloud

You'll end up with a permanent URL like `https://<username>-market-hub.streamlit.app` that doesn't rotate or expire.

## What's needed

- A GitHub account (free)
- A Streamlit Cloud account (free, signs in with GitHub)
- ~10 minutes

## Step 1 — push the code to GitHub

From this directory (`/Users/lj/market-hub`):

```bash
# Initialize git (only once)
git init
git add .
git status   # verify .streamlit/secrets.toml is NOT in the list

# If secrets.toml shows up, STOP — your .gitignore isn't working.
# Otherwise:
git commit -m "Initial commit"

# Create the repo on GitHub (web: github.com/new → name it "market-hub" → don't add README/license)
# Then push:
git remote add origin https://github.com/<your-username>/market-hub.git
git branch -M main
git push -u origin main
```

**Public vs private repo:** Streamlit Cloud free tier deploys from both. Public is fine — your secrets aren't in the repo (they go to Streamlit's secrets UI separately).

## Step 2 — deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **"Create app"** → **"Deploy a public app from GitHub"**
4. Pick your repo (`<username>/market-hub`)
5. Branch: `main`
6. **Main file path: `hub.py`** ← important
7. Python version: 3.11 (the default is fine)
8. Click **"Deploy"**

First build takes 3-5 minutes (installing pandas, yfinance, plotly, etc.). After that you'll see your app loading.

## Step 3 — paste secrets into Streamlit Cloud

On the deployed app's page, click **"Manage app"** (bottom-right) → **Settings** → **Secrets**.

Paste the entire contents of `.streamlit/secrets.toml.example` (already in the repo) and fill in just the keys you have. Minimum to make the app useful:

```toml
FINNHUB_KEY = "your_finnhub_key"
GEMINI_API_KEY = "your_gemini_key"
```

Save. The app auto-restarts and picks up the secrets.

## Step 4 — done

Your URL will be of the form `https://<username>-<repo>-hub-<hash>.streamlit.app` (Streamlit shows it on the app page). Bookmark it on your phone.

---

## Limitations vs running locally

| Feature | Local | Streamlit Cloud |
|---|---|---|
| Snapshot history persists across restarts | ✅ | ❌ wiped on restart |
| `picker.py` cron snapshots | ✅ via cron | ❌ no scheduler (see workaround below) |
| Alpaca paper trading | ✅ | ✅ |
| AI panel / Confluence vision | ✅ | ✅ |
| Heatmap / Backtester / News | ✅ | ✅ |
| Disk caches for fundamentals/insider data | persistent | session-only |
| App sleeps after inactivity | n/a | ⚠ goes to sleep after 7 days idle (auto-wakes on visit) |

## Workarounds for scheduled snapshots (optional)

If you want cron-driven snapshots in the cloud version, two paths:

**Option A — Keep cron on your Mac, sync via GitHub.**
On your Mac, `picker.py` runs Mondays and commits the new snapshot to a `snapshots` branch of the repo. Streamlit Cloud's app pulls the latest snapshots from a small endpoint. Adds complexity; only do this if you really want cloud-resident history.

**Option B — GitHub Actions cron.**
Move `picker.py` runs to a GitHub Actions workflow. The action commits each snapshot back to the repo. Secrets are managed via GitHub Actions secrets.

Skip both for now — the manual "Run + save snapshot" button in the Snapshots tab works fine for occasional use.

## Updating after the first deploy

After making local code changes:

```bash
git add .
git commit -m "what you changed"
git push
```

Streamlit Cloud auto-redeploys within ~30 seconds of the push.

## Troubleshooting

- **"App not found"** — repo is private and Streamlit Cloud doesn't have access. Re-authorize in Streamlit settings, or make the repo public.
- **"ModuleNotFoundError"** — package missing from `requirements.txt`. Add it, commit, push.
- **App crashes on startup** — click "Manage app" → check logs. Most common: missing/malformed secret. Re-check the Secrets pane.
- **"app.py not found alongside hub.py"** — both files must be at repo root. If you put them in a subdirectory, set the main file path accordingly.
