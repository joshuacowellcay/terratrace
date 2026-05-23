import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'TerraTrace - forest-loss change detection',
  description:
    'Draw a region, pick two dates, and see a per-pixel forest-loss heatmap plus a hectares-lost estimate.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
