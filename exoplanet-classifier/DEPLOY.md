Firebase hosting deployment (local)

1. Install the Firebase CLI if you don't have it:

   npm install -g firebase-tools

2. Log in and initialize (run once):

   firebase login
   cd exoplanet-classifier
   firebase init hosting

   - When prompted, select your Firebase project or create a new one.
   - Choose `build` as the public directory.
   - Configure as a single-page app (yes) so that index.html is served for all routes.

3. Build and deploy:

   npm run build
   firebase deploy --only hosting

CI (GitHub Actions)

- The included workflow will run on pushes to `master` and deploy using the
  provided GitHub secrets. You'll need to add the following repository secrets:

  - FIREBASE_SERVICE_ACCOUNT: JSON service account key with project deploy rights
  - FIREBASE_PROJECT_ID: your Firebase project id

See the workflow file at .github/workflows/firebase-deploy.yml for details.
