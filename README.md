# EchoReview — local and free deployment guide

Your users will use **one URL only**: the Cloudflare Pages website URL, for
example `https://your-project.pages.dev`. The Node API and AI service run in
the background as supporting services; users never need to open their URLs.

Reply generation is resilient:

```text
Browser → Node API → Groq AI service → Groq
                   ↘ local safe template fallback
```

If Groq errors, times out, runs out of quota, or the AI service is asleep/down,
the Node API returns a template reply within about nine seconds. The UI labels
the saved reply as **Generated with Groq AI** or **Template fallback used**.

## Before you start

You need accounts on GitHub, Cloudflare, Render, MongoDB Atlas, and Groq. They
all have free plans for personal/demo usage. Free services have limits: Render
web services sleep after 15 idle minutes, so a first request can be slow; Groq
also applies rate limits. No hosting choice can guarantee unlimited production
usage for $0.

Never commit a real `.env`, MongoDB password, JWT secret, or Groq key.

## Run locally first

Open three terminals. Every time you modify an `.env`, stop and restart the
corresponding service.

### Terminal 1: Python AI service

```bash
cd ~/Documents/code/GITHUB/Echoreview/ai-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `ai-service/.env` and put a **newly rotated** key here:

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TIMEOUT_SECONDS=6
CLIENT_ORIGINS=http://localhost:5173
```

Run and test it:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/health
```

### Terminal 2: Node API

First create a local MongoDB database, or use an Atlas connection string. Then:

```bash
cd ~/Documents/code/GITHUB/Echoreview/server
npm install
cp .env.example .env
```

Set at minimum in `server/.env`:

```env
PORT=5000
MONGODB_URI=mongodb://127.0.0.1:27017/echoreviewai
JWT_SECRET=replace-this-with-a-long-random-secret
JWT_EXPIRES=7d
AI_SERVICE_URL=http://127.0.0.1:8000
AI_SERVICE_TIMEOUT_MS=9000
CLIENT_ORIGINS=http://localhost:5173
```

Start it:

```bash
npm start
```

### Terminal 3: React client

```bash
cd ~/Documents/code/GITHUB/Echoreview/client
npm install
printf 'VITE_API_BASE_URL=http://localhost:5000/api\nVITE_AI_SERVICE_URL=http://localhost:8000\n' > .env.local
npm run dev
```

Open the Vite URL, normally http://localhost:5173. On **AI generated**, the
Node console should show `forwarding ... /generate-reply`; the Python console
should show `POST /generate-reply`. A failed Groq call still produces a reply
with the template label.

## Deploy for free and get one website URL

### 1. Push the project to GitHub

```bash
cd ~/Documents/code/GITHUB/Echoreview
git add .
git commit -m "Prepare free deployment"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/echoreview.git
git push -u origin main
```

Create the GitHub repository first. If `origin` already exists, use
`git remote set-url origin ...` instead of `git remote add`.

### 2. Create a free MongoDB Atlas database

1. In MongoDB Atlas, create an **M0 Free** cluster.
2. Create a database user with a strong password.
3. In **Network Access**, add `0.0.0.0/0`. This is needed because Render's
   outgoing IP is not fixed; use a strong database password.
4. Click **Connect → Drivers**, copy the `mongodb+srv://...` URI, and replace
   `<password>` with your encoded password.

Keep that URI ready as `MONGODB_URI`.

### 3. Deploy the AI service on Render

1. In Render, choose **New → Web Service**, connect the GitHub repository.
2. Use these values:

   | Setting | Value |
   | --- | --- |
   | Name | `echoreview-ai` |
   | Root Directory | `ai-service` |
   | Runtime | Python 3 |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
   | Instance Type | Free |

3. Add environment variables:

   ```text
   GROQ_API_KEY = your Groq key
   GROQ_MODEL = llama-3.3-70b-versatile
   GROQ_TIMEOUT_SECONDS = 6
   CLIENT_ORIGINS = https://placeholder.invalid
   ```

4. Deploy, wait for success, and copy its URL, such as
   `https://echoreview-ai.onrender.com`. Open `/health` to check it works.

### 4. Deploy the Node API on Render

Create another **Web Service** from the same repository:

| Setting | Value |
| --- | --- |
| Name | `echoreview-api` |
| Root Directory | `server` |
| Runtime | Node |
| Build Command | `npm ci` |
| Start Command | `npm start` |
| Instance Type | Free |

Add these environment variables:

```text
MONGODB_URI = your Atlas URI
JWT_SECRET = a long random string (at least 32 characters)
JWT_EXPIRES = 7d
AI_SERVICE_URL = https://echoreview-ai.onrender.com
AI_SERVICE_TIMEOUT_MS = 9000
CLIENT_ORIGINS = https://placeholder.invalid
```

Deploy and copy the API URL, such as `https://echoreview-api.onrender.com`.
Open `https://echoreview-api.onrender.com/api/health` and confirm JSON returns.

### 5. Deploy the frontend on Cloudflare Pages — this is the one URL you share

1. Cloudflare Dashboard → **Workers & Pages → Create → Pages → Connect to Git**.
2. Select the repository and use:

   | Setting | Value |
   | --- | --- |
   | Project name | `echoreview` |
   | Production branch | `main` |
   | Root directory | `client` |
   | Build command | `npm run build` |
   | Build output directory | `dist` |

3. Add production environment variables:

   ```text
   VITE_API_BASE_URL = https://echoreview-api.onrender.com/api
   VITE_AI_SERVICE_URL = https://echoreview-ai.onrender.com
   ```

4. Deploy. Copy the Pages URL, for example `https://echoreview.pages.dev`.
   **This is the single URL you give to users.**

### 6. Connect the final frontend URL

In both Render services, replace `CLIENT_ORIGINS` with the exact Cloudflare
Pages URL from step 5 (no trailing slash), for example:

```text
https://echoreview.pages.dev
```

Save/redeploy both services. This allows the dashboard’s health, sentiment,
insight, issue-map, and issue-summary calls to reach the AI service, while
reply generation continues to go securely through the API.

## End-to-end deployment check

1. Visit the Pages URL in a private/incognito window.
2. Log in and confirm the review list loads.
3. Click **AI generated** on a review.
4. Confirm the editor reads either **Generated with Groq AI** or **Template
   fallback used**, then save it.
5. Refresh and confirm the reply and label persist.
6. Stop/remove `GROQ_API_KEY` temporarily from the AI Render service and retry:
   the result must be **Template fallback used**. Restore the key afterward.

The template fallback lives in both the AI service and Node API. It does not
need a Groq key, quota, or an active AI service.
