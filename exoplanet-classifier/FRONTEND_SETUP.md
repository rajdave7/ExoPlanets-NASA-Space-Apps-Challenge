Exoplanet Classifier Frontend

This folder contains the React frontend for the Exoplanet Classifier.

Space-themed UI (Tailwind)

I updated the UI to use Tailwind CSS and a lightweight space background to give the app a space/astronomy vibe without changing any functionality.

Quick setup:

1. From this folder run (PowerShell):

```
npm install
npm start
```

2. The app will run at http://localhost:3000 by default (ensure backend FastAPI at http://localhost:8000 is running for full functionality).

Notes:
- Tailwind and PostCSS configs were added. If your environment already pins versions, run `npm install` to update the lockfile.
- The UI changes are purely presentational: fonts, background, and subtle nebula panels.
