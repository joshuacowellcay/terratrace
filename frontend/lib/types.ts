export type BBox = {
  west: number;
  south: number;
  east: number;
  north: number;
};

export type PredictRequest = {
  bbox: BBox;
  before_date: string;
  after_date: string;
  threshold?: number;
};

export type PredictResponse = {
  prediction_id: string;
  bbox: BBox;
  hectares_lost: number;
  pixels_loss: number;
  pixels_total: number;
  mean_confidence: number;
  cloud_fraction: number;
  source: 'real' | 'sample';
  notes: string;
};
