'use client';

import { useState } from 'react';
import { predict } from '@/lib/api';
import type { BBox, PredictResponse } from '@/lib/types';

type Props = {
  bbox: BBox | null;
  onPrediction: (p: PredictResponse) => void;
  overlay: 'mask' | 'confidence' | 'before' | 'after' | 'none';
  onOverlayChange: (kind: 'mask' | 'confidence' | 'before' | 'after' | 'none') => void;
  prediction: PredictResponse | null;
};

export default function PredictionPanel({
  bbox,
  onPrediction,
  overlay,
  onOverlayChange,
  prediction,
}: Props) {
  const [beforeDate, setBeforeDate] = useState('2019-07-15');
  const [afterDate, setAfterDate] = useState('2023-07-15');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async () => {
    if (!bbox) {
      setError('Click two points on the map to define an area first.');
      return;
    }
    if (!beforeDate || !afterDate) {
      setError('Both dates are required.');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await predict({ bbox, before_date: beforeDate, after_date: afterDate });
      onPrediction(res);
      onOverlayChange('mask');
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-full flex-col gap-6 overflow-y-auto bg-white p-6 text-canopy-900">
      <header>
        <h1 className="text-xl font-semibold text-canopy-900">TerraTrace</h1>
        <p className="text-sm text-canopy-700">
          Forest-loss change detection. Click two corners on the map to define an AOI, pick a
          before / after date, then compute.
        </p>
      </header>

      <section className="rounded-lg border border-canopy-100 bg-canopy-50 p-4 text-sm">
        <div className="font-medium text-canopy-700">Selected area</div>
        <div className="mt-1 font-mono text-xs text-canopy-900" data-testid="bbox-display">
          {bbox
            ? `W ${bbox.west.toFixed(4)}  S ${bbox.south.toFixed(4)}  E ${bbox.east.toFixed(4)}  N ${bbox.north.toFixed(4)}`
            : 'no AOI yet - click two corners on the map'}
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <label className="text-sm font-medium text-canopy-700">
          Before
          <input
            type="date"
            value={beforeDate}
            onChange={(e) => setBeforeDate(e.target.value)}
            className="mt-1 block w-full rounded border border-canopy-100 px-3 py-1.5 text-sm text-canopy-900 focus:outline-none focus:ring-2 focus:ring-canopy-500"
          />
        </label>
        <label className="text-sm font-medium text-canopy-700">
          After
          <input
            type="date"
            value={afterDate}
            onChange={(e) => setAfterDate(e.target.value)}
            className="mt-1 block w-full rounded border border-canopy-100 px-3 py-1.5 text-sm text-canopy-900 focus:outline-none focus:ring-2 focus:ring-canopy-500"
          />
        </label>
        <button
          type="button"
          onClick={onSubmit}
          disabled={loading || !bbox}
          className="mt-2 rounded bg-canopy-700 px-4 py-2 text-sm font-medium text-white hover:bg-canopy-900 disabled:cursor-not-allowed disabled:opacity-50"
          data-testid="compute-button"
        >
          {loading ? 'computing...' : 'Compute forest-loss'}
        </button>
        {error && (
          <p className="text-xs text-red-700" data-testid="error">
            {error}
          </p>
        )}
      </section>

      {prediction && (
        <section className="rounded-lg border border-canopy-100 p-4">
          <div className="flex items-baseline justify-between">
            <h2 className="text-sm font-semibold text-canopy-700">Result</h2>
            <span className="text-xs uppercase tracking-wide text-canopy-500">
              source: {prediction.source}
            </span>
          </div>
          <dl className="mt-3 grid grid-cols-2 gap-y-2 text-sm">
            <dt className="text-canopy-700">hectares lost</dt>
            <dd className="text-right font-mono text-canopy-900" data-testid="hectares">
              {prediction.hectares_lost.toFixed(1)} ha
            </dd>
            <dt className="text-canopy-700">% pixels lost</dt>
            <dd className="text-right font-mono text-canopy-900">
              {((100 * prediction.pixels_loss) / Math.max(1, prediction.pixels_total)).toFixed(1)}%
            </dd>
            <dt className="text-canopy-700">mean confidence</dt>
            <dd className="text-right font-mono text-canopy-900">
              {(100 * prediction.mean_confidence).toFixed(1)}%
            </dd>
            <dt className="text-canopy-700">cloud fraction</dt>
            <dd className="text-right font-mono text-canopy-900">
              {(100 * prediction.cloud_fraction).toFixed(1)}%
            </dd>
          </dl>
          {prediction.notes && (
            <p className="mt-3 rounded bg-canopy-50 p-2 text-xs text-canopy-700">
              {prediction.notes}
            </p>
          )}

          <div className="mt-4">
            <div className="text-xs font-medium uppercase tracking-wide text-canopy-500">
              overlay
            </div>
            <div className="mt-1 flex flex-wrap gap-1">
              {(['before', 'after', 'mask', 'confidence', 'none'] as const).map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => onOverlayChange(k)}
                  className={`rounded border px-2 py-1 text-xs ${
                    overlay === k
                      ? 'border-canopy-700 bg-canopy-700 text-white'
                      : 'border-canopy-100 text-canopy-700 hover:bg-canopy-50'
                  }`}
                  data-testid={`overlay-${k}`}
                >
                  {k}
                </button>
              ))}
            </div>
          </div>
        </section>
      )}

      <footer className="mt-auto text-xs text-canopy-700">
        <a
          href="https://github.com/joshuacowellcay/terratrace"
          className="underline hover:text-canopy-900"
          target="_blank"
          rel="noreferrer"
        >
          source on GitHub
        </a>
      </footer>
    </div>
  );
}
