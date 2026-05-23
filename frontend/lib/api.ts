import type { PredictRequest, PredictResponse } from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export async function predict(req: PredictRequest): Promise<PredictResponse> {
  const res = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === 'string') detail = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return (await res.json()) as PredictResponse;
}

export function tileImageUrl(
  predictionId: string,
  kind: 'before' | 'after' | 'mask' | 'confidence',
): string {
  return `${API_URL}/tile/${predictionId}/${kind}.png`;
}

export function apiBaseUrl(): string {
  return API_URL;
}
