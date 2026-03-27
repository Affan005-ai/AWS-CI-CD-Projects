# Student Math Score Predictor

# From Model to Production: Dual AWS Deployment (Elastic Beanstalk + Docker/ECR/EC2 CI/CD)

An end-to-end Machine Learning web application that predicts a student's math score using:

- Gender
- Race/Ethnicity
- Parental level of education
- Lunch type
- Test preparation course status
- Reading score
- Writing score

This repository is the **Docker + ECR + EC2 CI/CD deployment track** on AWS.
The Elastic Beanstalk track is maintained in a separate companion repository.

## Repository Map

1. Docker + ECR + EC2 CI/CD (this repo):
   - `https://github.com/Affan005-ai/AWS-CI-CD-Projects`
2. Elastic Beanstalk deployment repo:
   - `https://github.com/Affan005-ai/ML-Projects`

## Problem Statement

Given student profile and performance context, estimate expected math score as a regression task.  
The goal is not just model training, but full ML productization: data pipeline, model selection, web inference, and cloud deployment automation.

## Project Features

1. End-to-end ML pipeline from ingestion to model artifact creation
2. Flask web app for live prediction
3. Multi-model training with hyperparameter tuning
4. Two AWS deployment strategies for comparison
5. CI/CD automation with containerized delivery
6. Practical deployment debugging documented from real runs

## End-to-End Architecture

1. Data ingestion reads `Notebook/data/StudentsPerformance.csv`
2. Train-test split stored in `artifacts/train.csv` and `artifacts/test.csv`
3. Transformation pipeline builds:
   - Numeric preprocessing (median impute + standard scaling)
   - Ordinal encoding for education/lunch/test prep
   - One-hot encoding for nominal categorical features
4. Model training compares multiple regressors
5. Hyperparameter tuning runs using `RandomizedSearchCV`
6. Best model + preprocessor serialized to:
   - `artifacts/model_1.pkl`
   - `artifacts/preprocessor_1.pkl`
7. Flask app loads artifacts and serves inference on `/predict`
8. CI/CD builds Docker image and deploys to AWS

## ML Pipeline (Deep Dive)

### 1) Data Ingestion

Implemented in `src/components/data_ingestion.py`.

- Loads dataset
- Persists raw/train/test artifacts
- Uses deterministic split (`random_state=42`)

### 2) Data Transformation

Implemented in `src/components/data_transformation.py`.

- Numerical columns: `reading score`, `writing score`
- Ordinal columns:
  - `parental level of education`
  - `lunch`
  - `test preparation course`
- Nominal columns:
  - `gender`
  - `race/ethnicity`

Pipelines:

- Numeric: `SimpleImputer(strategy="median")` + `StandardScaler()`
- Ordinal: `SimpleImputer(strategy="most_frequent")` + `OrdinalEncoder(...)`
- Nominal: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`

### 3) Model Training and Selection

Implemented in `src/components/model_trainer.py`.

Models evaluated:

- RandomForestRegressor
- DecisionTreeRegressor
- GradientBoostingRegressor
- LinearRegression
- KNeighborsRegressor
- XGBRegressor
- AdaBoostRegressor

Selection rule:

- Best model selected by highest test R2 score
- Minimum quality threshold: `R2 >= 0.6`

### 4) Hyperparameter Tuning (How It Works Here)

Implemented in `src/utils.py` via `evaluate_models(...)`.

For each model with a parameter grid:

1. Run `RandomizedSearchCV` with:
   - `n_iter=9`
   - `cv=3`
   - `scoring="r2"`
   - `n_jobs=-1`
2. Pick best estimator from search
3. Evaluate on test set and record score
4. Compare all models and persist top performer

How to improve tuning further:

1. Increase `n_iter` (for wider search)
2. Add fixed `random_state` in search for reproducibility
3. Use nested CV for stronger model selection confidence
4. Log experiment runs to MLflow/W&B

## Inference Layer

`application.py` exposes:

- `/` -> form UI
- `/predict` -> receives features, applies preprocessor/model, returns score

Runtime binding:

- Flask runs on `0.0.0.0:5000` in this repository

## Deployment Track (Primary in This Repo): Docker + ECR + EC2 + GitHub Actions

Workflow file: `.github/workflows/main.yaml`

Pipeline jobs:

1. `integration`
2. `build-and-push-ecr-image`
3. `Continuous-Deployment` (self-hosted runner on EC2)

Required GitHub secrets:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_ECR_LOGIN_URI`
- `ECR_REPOSITORY_NAME`

EC2 setup for Docker runner:

```bash
sudo apt-get update -y
sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

Container deployment pattern used in CD:

```bash
docker pull $AWS_ECR_LOGIN_URI/$ECR_REPOSITORY_NAME:latest
docker rm -f mltest || true
docker run -d -p 8080:5000 --name mltest $AWS_ECR_LOGIN_URI/$ECR_REPOSITORY_NAME:latest
```

## Deployment Track (Related Repo): Elastic Beanstalk

Companion repository:

- `https://github.com/Affan005-ai/ML-Projects`

Relevant EB files used in that track:

- `.elasticbeanstalk/config.yml`
- `.ebextensions/python.config`
- `Procfile`

## Production-Grade Infrastructure Blueprint

Current repo is a strong learning-to-production bridge. For production hardening, apply:

1. Application Serving

- Replace Flask dev server with Gunicorn in Docker runtime
- Add health endpoint (`/health`)
- Set request timeout and worker count via environment variables

2. Security

- Move from long-lived IAM user keys to GitHub OIDC + IAM role
- Store secrets in AWS Secrets Manager or SSM Parameter Store
- Restrict EC2 Security Group ingress to required CIDRs

3. Scalability and Reliability

- Add Application Load Balancer in front of EC2
- Use Auto Scaling Group for runner/app instances (if traffic-based)
- Add rolling deployment strategy and rollback logic

4. Observability

- Centralize logs in CloudWatch
- Add container metrics and alarms (CPU, memory, disk, 5xx)
- Add deployment notifications (Slack/Teams/email)

5. CI/CD Quality Gates

- Replace placeholder lint/test with real checks
- Add vulnerability scan (Trivy/Grype) on image before push
- Add branch protection with required checks

## Real Issues Faced and Fixes

1. Docker container name conflict:

- Error: `container name "/mltest" is already in use`
- Fix: `docker rm -f mltest || true` before `docker run`

2. Port mismatch confusion:

- App listens inside container on `5000`
- Correct host mapping: `8080:5000` (not `8080:8080`)

3. Runner offline state:

- Self-hosted runner became offline after session reset
- Fix: run as service and check via `./svc.sh status`

## Repository Structure

```text
.
|-- .github/workflows/main.yaml
|-- .ebextensions/python.config
|-- .elasticbeanstalk/config.yml
|-- application.py
|-- Dockerfile
|-- Procfile
|-- src/
|   |-- components/
|   |-- pipeline/
|   `-- utils.py
|-- artifacts/
|-- templates/
|-- static/
`-- Notebook/data/
```

## Quick Start

```bash
pip install -r requirements.txt
python application.py
```

Open:

- `http://127.0.0.1:5000`

## Credits

This project was built as part of my ML engineering and deployment learning journey inspired by the teachings and project patterns shared by **Krish Naik**.

## Acknowledgment

Thanks to the open-source Python, scikit-learn, Flask, Docker, and AWS communities for documentation and tooling that made this implementation possible.
