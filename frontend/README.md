# TerraTrace frontend

Next.js 14 + TypeScript + Tailwind + MapLibre GL.

```bash
npm install
npm run dev        # http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev   # point at non-default backend
npm test
npm run build
```

The map calls the backend in `src/terratrace/inference/service.py` at `NEXT_PUBLIC_API_URL`
(default `http://localhost:8000`). Use the workspace `docker-compose.yml` at the repo root to
bring the backend up at the same time.
