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

## Before you start (first-time checklist)

You need accounts on GitHub, Cloudflare, Render, MongoDB Atlas, and Groq. They
all have free plans for personal/demo usage. Free services have limits: Render
web services sleep after 15 idle minutes, so a first request can be slow; Groq
also applies rate limits. No hosting choice can guarantee unlimited production
usage for $0.

Never commit a real `.env`, MongoDB password, JWT secret, or Groq key.

Create these free accounts in this order:

1. [GitHub](https://github.com) — stores the code that Render and Cloudflare deploy.
2. [MongoDB Atlas](https://www.mongodb.com/atlas/database) — stores users and reviews.
3. [Groq Console](https://console.groq.com/keys) — provides the AI key.
4. [Render](https://render.com) — runs the Python AI service and Node API.
5. [Cloudflare](https://dash.cloudflare.com/sign-up) — hosts the frontend and gives the single public website URL.

Use the same GitHub account to sign in to Render and Cloudflare when they ask
to connect a Git provider. You do not need a credit card for this guide.

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

You have already created and pushed the repository if you can open
`https://github.com/varshneyanushka/Echoreview`. In that case, skip this step.

Otherwise: go to GitHub → top-right **+** → **New repository**. Name it
`Echoreview`, choose **Public**, leave “Add a README” unchecked, and click
**Create repository**. Then copy the HTTPS repository URL and run:

```bash
cd ~/Documents/code/GITHUB/Echoreview
git add .
git commit -m "Prepare free deployment"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/Echoreview.git
git push -u origin main
```

If the terminal says `remote origin already exists`, do not run `git remote
add` again. Check it with `git remote -v`, then use `git push -u origin main`.

### 2. Create a free MongoDB Atlas database

1. Sign in at [cloud.mongodb.com](https://cloud.mongodb.com). Atlas may first
   show an onboarding screen; choose **Build a Database**.
2. Select the **Free** option, usually labelled **M0** or **Free/Sandbox**.
   Pick any suggested cloud provider and a nearby region, then click **Create**.
3. Wait for the cluster to finish creating. In the left navigation menu, open
   **Security** and click **Database Access**.
4. Click **Add New Database User**. Choose password authentication, enter a
   username and a long password, then click **Add User**. Save both somewhere
   private; you will need them once.
5. Still in the left navigation, under **Security**, click **Network Access**.
   This is where “Network Access” is located. Click **Add IP Address**.
6. In the popup, click **Allow Access from Anywhere**. Atlas fills in
   `0.0.0.0/0`; click **Confirm**. This permits Render to connect because a
   free Render service has no fixed outgoing IP. Keep the database user
   password strong and never place it in GitHub.
7. Go to **Database** in the left navigation, find your cluster, and click
   **Connect** → **Drivers**. Choose **Node.js** if asked. Copy the connection
   string beginning `mongodb+srv://`.
8. Replace `<db_password>` (or `<password>`) in that string with the password
   from step 4. If your password contains `@`, `:`, `/`, `?`, `#`, `[`, or `]`,
   URL-encode it first; easiest is to create a new database password using only
   letters, numbers, and `-` or `_`.

Keep that URI ready as `MONGODB_URI`.

### 3. Get a Groq key

1. Sign in at [console.groq.com/keys](https://console.groq.com/keys).
2. Click **Create API Key**, name it `echoreview`, copy it once, and keep it
   private. You will paste it into Render in step 4—not GitHub or Cloudflare.

### 4. Deploy the AI service on Render

1. Sign in at [dashboard.render.com](https://dashboard.render.com) with GitHub.
   Authorize access to the `Echoreview` repository when prompted.
2. Click **New +** (top-right) → **Web Service** → select `Echoreview` →
   **Connect**.
2. Use these values:

   | Setting | Value |
   | --- | --- |
   | Name | `echoreview-ai` |
   | Root Directory | `ai-service` |
   | Runtime | Python 3 |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
   | Instance Type | Free |

3. Scroll down to **Environment Variables** → **Add Environment Variable**.
   Add each row below, then choose the free instance type before creating:

   ```text
   GROQ_API_KEY = your Groq key
   GROQ_MODEL = llama-3.3-70b-versatile
   GROQ_TIMEOUT_SECONDS = 6
   CLIENT_ORIGINS = https://placeholder.invalid
   ```

4. Click **Create Web Service**. Wait for the log to say the service is live.
   Copy its URL from the top of the service page, such as
   `https://echoreview-ai.onrender.com`. Open `/health` to check it works.

### 5. Deploy the Node API on Render

1. In Render, click **New +** → **Web Service** again. Select the same
   `Echoreview` repository and click **Connect**.
2. Fill in the table below and add each environment variable in the same
   **Environment Variables** section:

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

3. Click **Create Web Service**. When it is live, copy the API URL, such as `https://echoreview-api.onrender.com`.
Open `https://echoreview-api.onrender.com/api/health` and confirm JSON returns.

### 6. Deploy the frontend on Cloudflare Pages — this is the one URL you share

1. Sign in to [Cloudflare Dashboard](https://dash.cloudflare.com), then select
   **Workers & Pages** in the left menu → **Create application** → **Pages** →
   **Connect to Git**.
2. Authorize GitHub if requested, choose `Echoreview`, then click **Begin setup**.
   Enter these values:

   | Setting | Value |
   | --- | --- |
   | Project name | `echoreview` |
   | Production branch | `main` |
   | Root directory | `client` |
   | Build command | `npm run build` |
   | Build output directory | `dist` |

3. Before clicking deploy, open **Environment variables (advanced)** → **Add
   variable**. Add both variables below to the **Production** environment:

   ```text
   VITE_API_BASE_URL = https://echoreview-api.onrender.com/api
   VITE_AI_SERVICE_URL = https://echoreview-ai.onrender.com
   ```

4. Click **Save and Deploy**. Wait until it completes, then copy the Pages URL,
   for example `https://echoreview.pages.dev`.
   **This is the single URL you give to users.**

### 7. Connect the final frontend URL

1. In Render, open `echoreview-ai` → **Environment** in the left sidebar.
   Edit `CLIENT_ORIGINS`, set it to the exact Cloudflare Pages URL from step 6
   (no trailing slash), then click **Save Changes**.
2. Repeat for `echoreview-api`.
3. Render redeploys after saving. Wait until both services are live again.

```text
https://echoreview.pages.dev
```

This allows the dashboard’s health, sentiment,
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
