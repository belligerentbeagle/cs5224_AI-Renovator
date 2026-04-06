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
