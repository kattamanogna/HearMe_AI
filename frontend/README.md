# Frontend (React + Vite)

Simple UI for submitting text, optional audio upload, and optional webcam face capture to:

`POST /api/v1/analyze`

## Run

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api/*` requests to `http://localhost:8000`.
