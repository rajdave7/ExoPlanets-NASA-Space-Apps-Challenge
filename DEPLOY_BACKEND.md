Deploying the FastAPI backend (container) to Azure Web App

Overview

- We'll build a Docker image for `backend.py` and deploy it to Azure Web App for Containers. The GitHub Actions workflow in `.github/workflows/azure-deploy-backend.yml` will build and push the container to your ACR and then update the Web App to use the image.

Prerequisites (Azure)

1. Azure subscription
2. Create a Resource Group
3. Create an Azure Container Registry (ACR)
   - Note the login server (e.g. myregistry.azurecr.io)
4. Create an Azure Web App for Containers (Linux) that will pull the image from ACR

Required GitHub repository secrets

- AZURE_CREDENTIALS : JSON for a service principal with contributor access (see `az ad sp create-for-rbac --sdk-auth`)
- ACR_LOGIN_SERVER : e.g. myregistry.azurecr.io
- ACR_REPOSITORY : e.g. exoplanet-backend
- WEBAPP_NAME : the name of the Azure Web App

Quick manual steps (one-time)

1. Build image locally and test:

   docker build -t exoplanet-backend:latest .
   docker run -p 8000:8000 exoplanet-backend:latest

   Then visit http://localhost:8000 to confirm the FastAPI app is running.

2. Create service principal and configure GitHub secrets (example):

   az ad sp create-for-rbac --name "github-actions-acr" --role contributor --scopes /subscriptions/<sub>/resourceGroups/<rg> --sdk-auth

   Copy the resulting JSON into the AZURE_CREDENTIALS secret.

Push to `master` to trigger the GitHub Actions workflow. The action will:

- build and push image to ACR
- update the Azure Web App to use the pushed image

If you prefer Firebase or another host (Render, Fly.io), tell me and I can create an alternate workflow.
