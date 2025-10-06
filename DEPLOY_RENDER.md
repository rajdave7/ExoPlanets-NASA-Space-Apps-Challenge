Deploying the FastAPI backend to Render (easy, free tier available)

Render is a simple alternative to Azure. It can build your Dockerfile from the repo and run the container with automatic deploys from your Git branch.

Steps:

1. Sign up for a free account at https://render.com and connect your GitHub account.
2. Click New -> Web Service.
3. Select this repository and the `master` branch.
4. For Environment choose `Docker` and set the Dockerfile path to `/Dockerfile`.
5. Set the Start Command to empty (Dockerfile CMD will run uvicorn on port 8000).
6. Set the PORT environment variable to `8000` in the Render service settings.
7. (Optional) Add any environment variables the backend needs.
8. Click Create Web Service — Render will build the image and deploy it. The generated public URL will be shown in the service dashboard.

Notes:

- This repo includes a `Dockerfile` and `render.yaml` manifest so Render can auto-create the service if you opt to import via the manifest.
- For persistent storage (models, uploaded_data), consider using an attached disk or a cloud storage bucket and update the `MODEL_DIR`/`DATA_DIR` paths in `backend.py` accordingly.
