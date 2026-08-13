# EchoReview

> **AI-powered customer reputation management and review intelligence platform**

EchoReview is a full-stack SaaS-style application that helps businesses collect, analyze, understand, and respond to customer reviews from a unified dashboard.

It combines a modern React frontend, a Node.js/Express application API, MongoDB persistence, and a dedicated Python/FastAPI AI service for sentiment analysis, issue detection, clustering, and actionable review insights.

---

## ✨ Highlights

- 📊 Review analytics dashboard
- 🧠 AI/ML-powered review intelligence
- 💬 AI-assisted review reply generation
- 🎯 Churn risk scoring from negative sentiment, low ratings, and unreplied negative reviews
- 📈 Sentiment velocity tracking
- 🗺️ PCA + clustering based Issue Cluster Map
- 🔎 Recurring issue/category analysis
- 🔐 JWT authentication
- 🗄️ MongoDB Atlas persistence
- ⚡ REST APIs
- 📡 Server-Sent Events for live analytics
- 🌐 Production deployment with Cloudflare + Render + MongoDB Atlas
- 🧩 Separate frontend, application API, and AI/ML service

---

# 🚀 Product Overview

Businesses receive customer reviews across platforms, but raw reviews are difficult to convert into operational decisions.

EchoReview transforms:

**Reviews → Sentiment → Issues → Clusters → Insights → Actions**

It helps answer:

- What are customers complaining about?
- Which issues are recurring?
- Is sentiment improving or declining?
- Which negative reviews remain unanswered?
- Which problems represent potential churn risk?
- What should the business prioritize?
- How can support teams respond faster?

---

# 🏗️ Architecture

```text
                         ┌─────────────────────────────┐
                         │       Customer / Admin       │
                         │          Browser             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                  ┌──────────────────────────────────────┐
                  │        Cloudflare Frontend            │
                  │      React + Vite + Tailwind          │
                  └──────────────┬───────────────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
                    ▼                          ▼
       ┌────────────────────────┐   ┌────────────────────────┐
       │    Node.js API         │   │    Python AI Service   │
       │    Express             │   │    FastAPI              │
       │                        │   │                        │
       │ Auth / JWT             │   │ Sentiment Analysis      │
       │ Reviews                │   │ Issue Analysis          │
       │ Analytics              │   │ PCA / Clustering        │
       │ Reply Gateway          │   │ AI Insights             │
       │ SSE                    │   │ ML Pipelines            │
       └───────────┬────────────┘   └────────────────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │      MongoDB Atlas      │
       │ users / reviews /       │
       │ categories / incomes /  │
       │ expenses                │
       └────────────────────────┘
```

### Architecture Philosophy

**Frontend** handles UI, dashboards, charts, authentication state, and API communication.

**Node.js API** handles authentication, JWT authorization, database access, review CRUD, analytics, reply-generation gateway logic, and live analytics.

**Python AI Service** isolates machine-learning and NLP workloads from the application server.

**MongoDB Atlas** provides persistent cloud storage.

This separation lets each layer evolve and scale independently.

---

# 🧰 Tech Stack

## Frontend

| Technology | Purpose |
|---|---|
| React 18 | UI framework |
| Vite | Build tooling |
| Tailwind CSS | Styling |
| Axios | HTTP communication |
| Recharts | Data visualization |
| JavaScript / JSX | Application logic |

## Backend

| Technology | Purpose |
|---|---|
| Node.js | Runtime |
| Express.js | REST API |
| MongoDB | Database |
| Mongoose | MongoDB ODM |
| JWT | Authentication |
| bcrypt/bcryptjs | Password security |
| Server-Sent Events | Live analytics |

## AI / ML

| Technology | Purpose |
|---|---|
| Python | AI/ML service |
| FastAPI | AI API layer |
| scikit-learn | ML, PCA, clustering |
| NumPy | Numerical processing |
| Hugging Face / Transformers | NLP model support |
| Custom sentiment model | Domain-specific sentiment analysis |
| Keyword analysis | Lightweight insights |
| Groq / configured LLM integrations | AI reply workflows where enabled |

## Infrastructure

| Technology | Role |
|---|---|
| Cloudflare | Frontend hosting |
| Render | Node.js API |
| Render | Python AI service |
| MongoDB Atlas | Cloud database |
| GitHub | Source control and deployment trigger |

---

# 📁 Repository Structure

```text
Echoreview/
│
├── client/
│   ├── src/
│   │   ├── components/
│   │   │   └── Dashboard.jsx
│   │   ├── api.js
│   │   └── ...
│   ├── package.json
│   ├── package-lock.json
│   └── ...
│
├── server/
│   ├── app.js
│   ├── models/
│   ├── routes/
│   ├── middleware/
│   └── ...
│
├── ai-service/
│   ├── main.py
│   ├── models/
│   │   └── sentiment_model/
│   ├── sentiment_model.py
│   ├── train.py
│   ├── train_reply.py
│   ├── requirements.txt
│   └── ...
│
└── README.md
```

---

# 🧠 AI & Analytics

## 1. Sentiment Analysis

Reviews are processed into Positive, Neutral, or Negative sentiment with a sentiment score, enabling customer perception monitoring at scale.

## 2. Churn Risk Score

EchoReview combines review signals into a business-oriented churn risk indicator:

```text
Churn Risk
    │
    ├── Negative review rate
    ├── Low-rating rate
    └── Unreplied negative reviews
```

## 3. Sentiment Velocity

Sentiment velocity compares earlier and recent review sentiment to show whether customer perception is improving or deteriorating.

```text
Earlier Reviews ───────────────► Recent Reviews
       42                              46
        │                               │
        └────────── +4 ────────────────┘
```

## 4. Issue Detection

Reviews can be grouped into recurring categories such as Delivery, Billing, Product, and General issues.

## 5. Issue Cluster Map

The dashboard combines engineered review features, dimensionality reduction, and clustering:

```text
Raw Reviews → Feature Engineering → PCA → Clustering → Issue Cluster Map
```

The resulting map exposes groups of reviews sharing similar characteristics across sentiment, rating, issue category, and platform.

## 6. AI Insights

The AI service exposes:

```text
POST /insights
```

and returns structured analytical data including:

```json
{
  "executiveSummary": "...",
  "insights": [],
  "recommendations": [],
  "stats": {},
  "clusters": [],
  "faultPatterns": [],
  "topComplaintTheme": "...",
  "generatedBy": "...",
  "generatedAt": "..."
}
```

---

# 💬 AI-Assisted Reply Generation

The frontend can request:

```text
POST /reviews/:reviewId/generate-reply
```

The Node.js server acts as the application gateway for the reply workflow. The response can be reviewed and saved against the customer review.

Keeping model-specific logic behind the backend prevents AI credentials and implementation details from being exposed to the browser.

---

# 🔐 Authentication

EchoReview uses JWT-based authentication.

```text
Login
  │
  ▼
POST /auth/login
  │
  ▼
JWT generated
  │
  ▼
Stored by frontend
  │
  ▼
Authorization: Bearer <token>
  │
  ▼
Protected API endpoints
```

The frontend Axios instance automatically attaches the JWT to authenticated API requests.

---

# 📡 Real-Time Analytics

The application supports live analytics using Server-Sent Events (SSE).

```text
Browser
   │
   │ persistent connection
   ▼
/analytics/stream
   │
   ▼
Node.js Server
   │
   ▼
Updated analytics
```

---

# 🌐 Production Deployment

### Frontend — Cloudflare

```text
https://echoreview.anushka-pkg.workers.dev/
```

### Application API — Render

```text
https://echoreview-api.onrender.com/
```

### AI Service — Render

```text
https://echoreview-ai.onrender.com/
```

### Database — MongoDB Atlas

```text
Cluster0
├── users
├── reviews
├── categories
├── incomes
└── expenses
```

---

# ⚙️ Production Configuration

## Frontend Environment Variables

```env
VITE_API_BASE_URL=https://echoreview-api.onrender.com/api
VITE_AI_SERVICE_URL=https://echoreview-ai.onrender.com
```

Vite environment variables are embedded during the production build, so changing them requires a new frontend build/deployment.

### Cloudflare Build Configuration

```text
Root directory: client
Build command: npm run build
Build output directory: dist
```

## Node.js API

Typical production variables:

```env
MONGODB_URI=<mongodb-atlas-connection-string>
JWT_SECRET=<strong-secret>
PORT=<render-provided-port>
```

## AI Service

Depending on enabled functionality:

```env
GROQ_API_KEY=<optional-key>
GEMINI_MODEL=<configured-model>
CUSTOM_MODEL_DIR=models/sentiment_model
SENTIMENT_MODEL=<configured-model>
```

The `/insights` endpoint can operate through the Python analytics pipeline without requiring an external LLM for every request.

---

# 🔄 End-to-End AI Insights Flow

```text
User clicks "AI Insights"
          │
          ▼
React Dashboard
          │
          ▼
fetchInsights(reviews)
          │
          ▼
POST /insights
          │
          ▼
FastAPI
          │
          ▼
Insight / clustering pipeline
          │
          ├── sentiment statistics
          ├── issue analysis
          ├── clustering
          ├── recommendations
          └── executive summary
          │
          ▼
Structured JSON
          │
          ▼
React dashboard
          │
          ▼
Business-facing AI Insights
```

---

# 🗄️ Database

MongoDB Atlas stores persistent application data.

Current collections include:

```text
users
reviews
categories
incomes
expenses
```

Seed data can be loaded into the cloud database so the production dashboard is immediately demonstrable.

---

# 🛠️ Local Development

## Clone

```bash
git clone https://github.com/varshneyanushka/Echoreview.git
cd Echoreview
```

## Frontend

```bash
cd client
npm install
npm run dev
```

Production build:

```bash
npm run build
```

## Node.js API

```bash
cd server
npm install
npm start
```

## AI Service

```bash
cd ai-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

---

# 🧪 Testing the AI Service

Health check:

```bash
curl https://echoreview-ai.onrender.com/health
```

Insights:

```bash
curl -X POST https://echoreview-ai.onrender.com/insights \
  -H "Content-Type: application/json" \
  -d '{"reviews":[]}'
```

---

# 🌍 CORS

Because the frontend and AI service use different origins, browser requests require CORS configuration.

Production frontend origin:

```text
https://echoreview.anushka-pkg.workers.dev
```

Example FastAPI configuration:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://echoreview.anushka-pkg.workers.dev",
        "https://echoreview.pages.dev",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

A direct `curl` request can succeed while a browser request fails because CORS is enforced by browsers.

---

# 🔒 Security

Never commit:

```text
.env
MongoDB credentials
API keys
JWT secrets
private credentials
```

Recommended practices:

- Store secrets in Render/Cloudflare environment variables
- Use strong JWT secrets
- Restrict MongoDB Atlas access appropriately
- Validate API input
- Protect authenticated routes
- Configure CORS explicitly
- Keep AI credentials server-side

---

# 📈 Scalability

The service-oriented architecture allows components to scale independently.

```text
Frontend traffic increases → Scale frontend
API traffic increases      → Scale Node.js service
ML workload increases      → Scale AI service
```

The AI layer can evolve without requiring a complete frontend rewrite.

---

# 💡 Engineering Decisions

### React + Vite
Fast component-based development and efficient production builds.

### Node.js + Express
A lightweight application API for authentication, CRUD, analytics, and AI gateway workflows.

### MongoDB
A natural fit for flexible review documents containing platform, sentiment, category, rating, reply, and timestamp metadata.

### Separate Python AI Service
Python provides a strong NLP/ML ecosystem while isolating ML dependencies from the application server.

### Server-Sent Events
A simple fit for predominantly server-to-client live analytics updates.

### Cloudflare + Render
Separates frontend delivery from API and ML compute, improving maintainability and independent scaling.

---

# 📊 Example Business Workflow

```text
Customer leaves review
        │
        ▼
Review stored in MongoDB
        │
        ▼
Sentiment / issue analysis
        │
        ▼
Dashboard updated
        │
        ├───────────────┐
        ▼               ▼
AI Insights        Churn Risk
        │               │
        └───────┬───────┘
                ▼
       Business prioritizes
          critical issues
                │
                ▼
        AI-assisted reply
                │
                ▼
        Response saved
```

---

# 🌟 What Makes EchoReview Stand Out

EchoReview is intentionally more than a CRUD review dashboard.

It demonstrates the integration of:

- Full-stack web development
- REST API design
- Authentication and authorization
- Database modeling
- Cloud deployment
- NLP
- Sentiment analysis
- Feature engineering
- PCA
- Clustering
- Business analytics
- AI-assisted workflows
- Real-time browser communication
- Service-oriented architecture

The project connects **machine-learning output to business decisions** rather than presenting ML as an isolated model demo.

---

# 🔭 Future Roadmap

Potential extensions:

- Multi-platform review ingestion
- Automated review synchronization
- Semantic embeddings
- Vector database integration
- Retrieval-augmented response generation
- Advanced topic modeling
- Time-series forecasting
- Automated escalation workflows
- Slack/email notifications
- Role-based access control
- Multi-tenant SaaS architecture
- Model monitoring and evaluation
- Human feedback loops for reply quality
- A/B testing of response strategies

---

# 🏆 Project Summary

**EchoReview** demonstrates how a modern full-stack system can combine application engineering, cloud infrastructure, and machine learning into a business-oriented product.

Instead of only asking:

> **"Is this review positive or negative?"**

EchoReview moves toward:

> **"What are customers telling us, what problems are recurring, what risks are emerging, and what should the business do next?"**

It converts customer feedback into measurable signals, recurring issue patterns, actionable insights, risk indicators, and response workflows.

---

# 🔗 Links

**GitHub Repository:** https://github.com/varshneyanushka/Echoreview  
**Live Frontend:** https://echoreview.anushka-pkg.workers.dev/  
**Application API:** https://echoreview-api.onrender.com/  
**AI Service:** https://echoreview-ai.onrender.com/

---

## 📜 License

Add the project's chosen license here if/when one is defined.

---

<p align="center">
  <b>EchoReview</b><br>
  Turning customer feedback into actionable intelligence.
</p>
