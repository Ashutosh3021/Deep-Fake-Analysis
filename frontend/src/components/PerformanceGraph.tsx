import { useEffect, useRef, useState } from 'react';
import { useGetDayRecords, useGetXPState } from '../hooks/useQueries';
import { Button } from '@/components/ui/button';

interface PerformanceGraphProps {
  fullView?: boolean;
}

export default function PerformanceGraph({ fullView = false }: PerformanceGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { data: dayRecords = [] } = useGetDayRecords();
  const { data: xpState } = useGetXPState();
  const [showProjection, setShowProjection] = useState(true);
  const [overlayMode, setOverlayMode] = useState<'none' | 'planned-vs-actual'>('none');

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = 40;

    // Clear canvas
    ctx.fillStyle = '#0B0E11';
    ctx.fillRect(0, 0, width, height);

    // Calculate performance scores
    const maxDays = 100;
    const performanceData: number[] = [];
    const plannedData: number[] = [];
    
    for (let i = 0; i < Math.min(maxDays, dayRecords.length); i++) {
      const record = dayRecords[i];
      const score = record.plannedXP > 0 
        ? (record.completedXP / record.plannedXP) * 100 
        : 0;
      performanceData.push(Math.min(100, Math.max(0, score)));
      plannedData.push(100); // Planned is always 100%
    }

    if (performanceData.length === 0) {
      ctx.fillStyle = '#8A8F98';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('NO PERFORMANCE DATA', width / 2, height / 2);
      return;
    }

    // Draw rank bands (subtle horizontal lines)
    const rankBands = [
      { y: 0.2, label: 'EXPERT' },
      { y: 0.4, label: 'ADVANCED' },
      { y: 0.6, label: 'INTERMEDIATE' },
      { y: 0.8, label: 'NOVICE' },
    ];

    ctx.strokeStyle = '#8A8F98';
    ctx.globalAlpha = 0.1;
    ctx.lineWidth = 1;

    rankBands.forEach(band => {
      const y = padding + (height - padding * 2) * band.y;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    });

    ctx.globalAlpha = 1;

    // Draw axes
    ctx.strokeStyle = '#8A8F98';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    const xStep = (width - padding * 2) / (maxDays - 1);
    const yScale = (height - padding * 2) / 100;

    // Draw planned vs actual overlay
    if (overlayMode === 'planned-vs-actual') {
      ctx.strokeStyle = '#8A8F98';
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.3;
      ctx.beginPath();

      plannedData.forEach((score, index) => {
        const x = padding + index * xStep;
        const y = height - padding - score * yScale;

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Draw performance line with EMA smoothing
    if (performanceData.length > 1) {
      const smoothedData: number[] = [];
      const alpha = 0.3; // EMA smoothing factor
      smoothedData[0] = performanceData[0];

      for (let i = 1; i < performanceData.length; i++) {
        smoothedData[i] = alpha * performanceData[i] + (1 - alpha) * smoothedData[i - 1];
      }

      ctx.strokeStyle = '#B6FF3B';
      ctx.lineWidth = 2;
      ctx.beginPath();

      smoothedData.forEach((score, index) => {
        const x = padding + index * xStep;
        const y = height - padding - score * yScale;

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();

      // Emphasize downward slopes
      ctx.strokeStyle = '#DC5F5F';
      ctx.lineWidth = 3;
      ctx.globalAlpha = 0.5;

      for (let i = 1; i < smoothedData.length; i++) {
        if (smoothedData[i] < smoothedData[i - 1]) {
          const x1 = padding + (i - 1) * xStep;
          const y1 = height - padding - smoothedData[i - 1] * yScale;
          const x2 = padding + i * xStep;
          const y2 = height - padding - smoothedData[i] * yScale;

          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }

      ctx.globalAlpha = 1;

      // Draw rank trajectory projection
      if (showProjection && smoothedData.length >= 7) {
        const recentData = smoothedData.slice(-7);
        const trend = (recentData[recentData.length - 1] - recentData[0]) / recentData.length;
        
        ctx.strokeStyle = '#B6FF3B';
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.3;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();

        const projectionDays = Math.min(14, maxDays - smoothedData.length);
        for (let i = 0; i <= projectionDays; i++) {
          const projectedScore = smoothedData[smoothedData.length - 1] + trend * i;
          const x = padding + (smoothedData.length - 1 + i) * xStep;
          const y = height - padding - Math.max(0, Math.min(100, projectedScore)) * yScale;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }

        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
    }

    // Draw labels
    ctx.fillStyle = '#8A8F98';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('DAY 1', padding, height - padding + 20);
    ctx.fillText(`DAY ${maxDays}`, width - padding, height - padding + 20);

    ctx.textAlign = 'right';
    ctx.fillText('100%', padding - 10, padding);
    ctx.fillText('0%', padding - 10, height - padding);

    // Warm-up indicator
    if (dayRecords.length <= 7) {
      ctx.fillStyle = '#B6FF3B';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('WARM-UP BUFFER', padding + 10, padding + 20);
    }

  }, [dayRecords, showProjection, overlayMode]);

  return (
    <div className="space-y-4 border border-[#8A8F98]/20 bg-[#0B0E11] p-6">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold text-[#E6E6E6]">100-DAY PERFORMANCE ANALYSIS</h2>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant={showProjection ? 'default' : 'outline'}
            onClick={() => setShowProjection(!showProjection)}
            className="border border-[#8A8F98]/20 bg-transparent text-[#8A8F98] hover:bg-[#8A8F98]/10"
          >
            PROJECTION
          </Button>
          <Button
            size="sm"
            variant={overlayMode === 'planned-vs-actual' ? 'default' : 'outline'}
            onClick={() => setOverlayMode(overlayMode === 'none' ? 'planned-vs-actual' : 'none')}
            className="border border-[#8A8F98]/20 bg-transparent text-[#8A8F98] hover:bg-[#8A8F98]/10"
          >
            PLANNED VS ACTUAL
          </Button>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        className="w-full"
        style={{ height: fullView ? '500px' : '300px' }}
      />
      <div className="text-xs text-[#8A8F98]">
        Performance normalized to completion rate. Downward trends emphasized.
        {showProjection && ' Faint projection line shows rank trajectory.'}
      </div>
    </div>
  );
}
