'use client';

import maplibregl, { Map as MaplibreMap } from 'maplibre-gl';
import { useEffect, useRef } from 'react';
import type { BBox, PredictResponse } from '@/lib/types';
import { tileImageUrl } from '@/lib/api';

type Props = {
  bbox: BBox | null;
  onBboxChange: (b: BBox) => void;
  prediction: PredictResponse | null;
  overlay: 'mask' | 'confidence' | 'before' | 'after' | 'none';
};

// Default centre: a deforestation hotspot in Rondonia, Brazil.
const DEFAULT_CENTER: [number, number] = [-63.05, -10.35];
const DEFAULT_ZOOM = 9;

// Free, no-key vector basemap from MapLibre's demo tiles. Replace with an own style URL for prod.
const BASEMAP_STYLE = 'https://demotiles.maplibre.org/style.json';

export default function Map({ bbox, onBboxChange, prediction, overlay }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MaplibreMap | null>(null);
  const firstCornerRef = useRef<{ lng: number; lat: number } | null>(null);

  // Initialise the map once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASEMAP_STYLE,
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      attributionControl: { compact: true },
    });
    mapRef.current = map;
    map.on('load', () => {
      // Empty AOI polygon source we'll populate as the user clicks corners.
      map.addSource('aoi', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      map.addLayer({
        id: 'aoi-fill',
        type: 'fill',
        source: 'aoi',
        paint: { 'fill-color': '#3c8d40', 'fill-opacity': 0.18 },
      });
      map.addLayer({
        id: 'aoi-outline',
        type: 'line',
        source: 'aoi',
        paint: { 'line-color': '#1f5a23', 'line-width': 2 },
      });
    });

    // Two-click bbox capture: first click drops a corner, second click closes the rectangle.
    map.on('click', (e) => {
      if (!firstCornerRef.current) {
        firstCornerRef.current = e.lngLat;
        return;
      }
      const c1 = firstCornerRef.current;
      const c2 = e.lngLat;
      firstCornerRef.current = null;
      const next: BBox = {
        west: Math.min(c1.lng, c2.lng),
        east: Math.max(c1.lng, c2.lng),
        south: Math.min(c1.lat, c2.lat),
        north: Math.max(c1.lat, c2.lat),
      };
      onBboxChange(next);
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
    // We deliberately initialise once; subsequent updates are handled by the effects below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Redraw the AOI polygon whenever the bbox changes.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const draw = () => {
      const source = map.getSource('aoi') as maplibregl.GeoJSONSource | undefined;
      if (!source) return;
      if (!bbox) {
        source.setData({ type: 'FeatureCollection', features: [] });
        return;
      }
      const poly = {
        type: 'Feature' as const,
        properties: {},
        geometry: {
          type: 'Polygon' as const,
          coordinates: [
            [
              [bbox.west, bbox.south],
              [bbox.east, bbox.south],
              [bbox.east, bbox.north],
              [bbox.west, bbox.north],
              [bbox.west, bbox.south],
            ],
          ],
        },
      };
      source.setData({ type: 'FeatureCollection', features: [poly] });
      map.fitBounds(
        [
          [bbox.west, bbox.south],
          [bbox.east, bbox.north],
        ],
        { padding: 80, maxZoom: 12, duration: 600 },
      );
    };
    if (map.isStyleLoaded()) draw();
    else map.once('load', draw);
  }, [bbox]);

  // Add / update the prediction overlay (before/after/mask/confidence).
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const sync = () => {
      // Remove previous overlay layers + source.
      if (map.getLayer('overlay-layer')) map.removeLayer('overlay-layer');
      if (map.getSource('overlay-source')) map.removeSource('overlay-source');

      if (!prediction || overlay === 'none') return;
      const b = prediction.bbox;
      map.addSource('overlay-source', {
        type: 'image',
        url: tileImageUrl(prediction.prediction_id, overlay),
        coordinates: [
          [b.west, b.north],
          [b.east, b.north],
          [b.east, b.south],
          [b.west, b.south],
        ],
      });
      map.addLayer({
        id: 'overlay-layer',
        type: 'raster',
        source: 'overlay-source',
        paint: { 'raster-opacity': overlay === 'mask' ? 0.85 : 0.7 },
      });
    };
    if (map.isStyleLoaded()) sync();
    else map.once('load', sync);
  }, [prediction, overlay]);

  return (
    <div ref={containerRef} className="absolute inset-0" data-testid="map-container" />
  );
}
