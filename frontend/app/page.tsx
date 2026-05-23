'use client';

import dynamic from 'next/dynamic';
import { useState } from 'react';
import PredictionPanel from '@/components/PredictionPanel';
import type { BBox, PredictResponse } from '@/lib/types';

// MapLibre needs `window`, so load the map client-side only and keep the initial server
// render free of browser globals.
const Map = dynamic(() => import('@/components/Map'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center bg-canopy-900 text-canopy-100">
      loading map...
    </div>
  ),
});

export default function HomePage() {
  const [bbox, setBbox] = useState<BBox | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [overlay, setOverlay] = useState<'mask' | 'confidence' | 'before' | 'after' | 'none'>(
    'none',
  );

  return (
    <main className="flex h-screen w-screen">
      <div className="relative w-2/3">
        <Map bbox={bbox} onBboxChange={setBbox} prediction={prediction} overlay={overlay} />
      </div>
      <aside className="w-1/3 min-w-[320px] border-l border-canopy-100 bg-white">
        <PredictionPanel
          bbox={bbox}
          onPrediction={setPrediction}
          overlay={overlay}
          onOverlayChange={setOverlay}
          prediction={prediction}
        />
      </aside>
    </main>
  );
}
