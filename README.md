# RoomStyle AI — Developer Guide

## Local Development

Two servers run in parallel: the **FastAPI backend** (uvicorn) and the **React frontend** (Vite).

### Prerequisites

| Tool | Install |
|------|---------|
| Python 3.12 | `pyenv install 3.12.0` |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | `pip install uv` or `brew install uv` |
| Node.js 18+ | `brew install node` or [nodejs.org](https://nodejs.org) |

---

## Backend

### 1. Install Python dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in `.env` — minimum required to run without a database:

```
GEMINI_API_KEY=your-gemini-api-key-here
DATABASE_URL=postgresql://roomstyle_admin:PASSWORD@<RDS_ENDPOINT>:5432/roomstyle
JWT_SECRET=any-random-string-for-local-dev
AWS_REGION=ap-southeast-1
S3_BUCKET=roomstyle-cs5224
```

> **No DB yet?** The `/products` and `/products/from-url` endpoints work without a database (they hit the IKEA API directly). Auth, projects, and generation require `DATABASE_URL` to be set to a live RDS instance.

### 3. Apply the database schema (first time only)

```bash
psql -h <RDS_ENDPOINT> -U roomstyle_admin -d roomstyle -f schema.sql
```

### 4. Run the backend

```bash
uv run uvicorn app.main:app --reload
```

API available at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`

---

## Frontend

### 1. Install Node dependencies

```bash
cd frontend
npm install
```

### 2. Configure environment

```bash
cp .env.example .env
```

The default `.env` points to the local backend:

```
VITE_API_URL=http://localhost:8000
```

For the deployed Lambda backend, change this to the API Gateway URL:

```
VITE_API_URL=https://dopwgelhc8.execute-api.ap-southeast-1.amazonaws.com/default/roomstyle-backend-api
```

### 3. Run the frontend

```bash
npm run dev
```

Frontend available at `http://localhost:5173`

---

## Running both together (recommended)

Open two terminal tabs from the project root:

**Tab 1 — Backend:**
```bash
uv run uvicorn app.main:app --reload
```

**Tab 2 — Frontend:**
```bash
cd frontend && npm run dev
```

Then open `http://localhost:5173`.

---

## What works without a database

| Feature | Needs DB? |
|---------|-----------|
| Browse Furniture (IKEA search) | No |
| URL indexer (`POST /products/from-url`) | No |
| Sign in / Register | Yes |
| My Projects | Yes |
| Generate AI Design | Yes |

---

## Key API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/products?q=sofa&source=ikea` | Search IKEA products |
| `GET` | `/products/{id}` | Get product detail |
| `POST` | `/products/from-url` | Scrape + normalise any furniture URL |
| `POST` | `/auth/register` | Create account |
| `POST` | `/auth/login` | Sign in, returns JWT |
| `POST` | `/projects` | Create project (auth required) |
| `POST` | `/generate/room` | Start generation job (auth required) |
| `GET` | `/generations/{id}` | Poll generation status |
| `GET` | `/health` | Health check |

#### POST /products/from-url

```json
{ "url": "https://www.ikea.com/us/en/p/klippan-loveseat-vissle-gray-s09010617/" }
```

Tries JSON-LD / OpenGraph first. Falls back to Gemini LLM extraction if structured data is incomplete. Returns a `ScrapedProduct`.

---

# RoomStyle: AWS Infrastructure Overview (Phase 1)

This document summarizes the initial cloud environment established for the RoomStyle project in the **Asia Pacific (Singapore) `ap-southeast-1`** region.

---

## 1. Team Access and IAM

**User Names:** ethan.wei, cleon.tan, anant.shanker, kaitlyn.goh, yash.mishra, sithanathan.rahul

| Field              | Value                                            |
| ------------------ | ------------------------------------------------ |
| Login URL          | https://cs5224-nus.signin.aws.amazon.com/console |
| Temporary Password | `RoomStyle2026!`                                 |

> **Note:** All team members must ensure they are operating within the **Singapore (`ap-southeast-1`)** region to maintain consistency with the provisioned resources.

---

## 2. Storage (Amazon S3)

The system uses a decoupled storage layer for managing room imagery.

| Field       | Value              |
| ----------- | ------------------ |
| Bucket Name | `roomstyle-cs5224` |

**Bucket Structure:**

| Folder         | Purpose                                                          |
| -------------- | ---------------------------------------------------------------- |
| `uploads/`     | Stores raw room photos uploaded by users for design generation   |
| `generations/` | Stores final AI-rendered interior designs and project blueprints |

---

## 3. Database (Amazon RDS PostgreSQL)

A relational database handles structured application data including users, projects, and design results.

| Field              | Value             |
| ------------------ | ----------------- |
| Database Name      | `roomstyle_db`    |
| Master Username    | `postgres_admin`  |
| Master Password    | `RoomStyle2026!`  |
| VPC Security Group | `roomstyle-db-sg` |

**Implementation:** Provisioned as a PostgreSQL instance to support the Fiscal Architect and Global Product Catalogue features.

---

## 4. Backend (Lambda and API Gateway)

The backend follows a serverless architecture using **FastAPI** and **Mangum**.

| Field        | Value                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------- |
| Lambda Name  | `roomstyle-backend-api`                                                                     |
| Runtime      | Python 3.12                                                                                 |
| Architecture | `arm64`                                                                                     |
| Trigger      | HTTP API                                                                                    |
| API Endpoint | `https://dopwgelhc8.execute-api.ap-southeast-1.amazonaws.com/default/roomstyle-backend-api` |

**Logic Flow:** The Lambda function acts as an orchestrator for furniture searches and room generation requests.

---

## Future Changes: Phase 2 Hardening

The current setup is optimised for rapid prototyping and cost-efficiency. The following changes are planned for the next phase.

### 1. Security and Authentication

- **API Gateway:** Transition from open access to JWT or IAM-based authorisation to secure the primary AI generation workflow.
- **Secrets Management:** Move database credentials and external API keys (IKEA, Taobao) into AWS Secrets Manager.
- **S3 Security:** Implement pre-signed URLs to allow secure, direct uploads from the frontend while keeping the bucket private.

### 2. Networking (VPC Strategy)

- **Private Subnets:** Relocate the RDS instance and future AI rendering engines into private subnets for enhanced security.
- **VPC Endpoints:** Establish gateway endpoints for S3 to enable internal service communication without traversing the public internet.

### 3. AI Engine and Job Queue

- **EC2 G5 Provisioning:** Deploy the AI Rendering Engine using Stable Diffusion and ControlNet on GPU-optimised instances.
- **SQS Integration:** Utilise Amazon SQS to manage the design job queue, ensuring the system can handle bursty image-generation workloads.

### 4. CI/CD and Frontend Integration

- **AWS Amplify:** Connect the GitHub repository to AWS Amplify to enable automated frontend hosting and content delivery.
- **Dependency Management:** Use Poetry to lock environment versions and ensure consistency across development and production Lambda environments.


## Database Intricacies

RDS Proxy (important since we're using Lambda):
Lambda functions can spin up many concurrent connections and exhaust your database's connection limit. Go to RDS → Proxies → Create proxy, point it at your RDS instance, and configure it to pull credentials from Secrets Manager. Your Lambda functions then connect to the proxy endpoint instead of the database directly.