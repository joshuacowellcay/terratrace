import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import PredictionPanel from '../PredictionPanel';
import type { BBox, PredictResponse } from '@/lib/types';

jest.mock('@/lib/api', () => ({
  predict: jest.fn(),
  tileImageUrl: jest.fn(),
  apiBaseUrl: () => 'http://localhost:8000',
}));

import { predict } from '@/lib/api';

const TEST_BBOX: BBox = { west: -63.1, south: -10.4, east: -63.0, north: -10.3 };

const TEST_RESPONSE: PredictResponse = {
  prediction_id: 'abc123',
  bbox: TEST_BBOX,
  hectares_lost: 12.5,
  pixels_loss: 1250,
  pixels_total: 16384,
  mean_confidence: 0.91,
  cloud_fraction: 0.05,
  source: 'sample',
  notes: 'unit test',
};

describe('PredictionPanel', () => {
  beforeEach(() => {
    (predict as jest.Mock).mockReset();
  });

  test('shows AOI placeholder when no bbox is selected', () => {
    render(
      <PredictionPanel
        bbox={null}
        onPrediction={() => {}}
        overlay="none"
        onOverlayChange={() => {}}
        prediction={null}
      />,
    );
    expect(screen.getByTestId('bbox-display').textContent).toMatch(/no AOI yet/);
  });

  test('Compute button is disabled until a bbox exists', () => {
    render(
      <PredictionPanel
        bbox={null}
        onPrediction={() => {}}
        overlay="none"
        onOverlayChange={() => {}}
        prediction={null}
      />,
    );
    const btn = screen.getByTestId('compute-button') as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });

  test('submits and renders prediction results', async () => {
    (predict as jest.Mock).mockResolvedValueOnce(TEST_RESPONSE);
    const onPrediction = jest.fn();
    const onOverlayChange = jest.fn();
    const { rerender } = render(
      <PredictionPanel
        bbox={TEST_BBOX}
        onPrediction={onPrediction}
        overlay="none"
        onOverlayChange={onOverlayChange}
        prediction={null}
      />,
    );
    fireEvent.click(screen.getByTestId('compute-button'));
    await waitFor(() => expect(onPrediction).toHaveBeenCalledWith(TEST_RESPONSE));
    expect(onOverlayChange).toHaveBeenCalledWith('mask');

    // Simulate parent rerendering with the prediction prop set.
    rerender(
      <PredictionPanel
        bbox={TEST_BBOX}
        onPrediction={onPrediction}
        overlay="mask"
        onOverlayChange={onOverlayChange}
        prediction={TEST_RESPONSE}
      />,
    );
    expect(screen.getByTestId('hectares').textContent).toContain('12.5 ha');
  });

  test('surfaces error message when the API fails', async () => {
    (predict as jest.Mock).mockRejectedValueOnce(new Error('too cloudy'));
    render(
      <PredictionPanel
        bbox={TEST_BBOX}
        onPrediction={() => {}}
        overlay="none"
        onOverlayChange={() => {}}
        prediction={null}
      />,
    );
    fireEvent.click(screen.getByTestId('compute-button'));
    await waitFor(() => expect(screen.getByTestId('error').textContent).toBe('too cloudy'));
  });
});
