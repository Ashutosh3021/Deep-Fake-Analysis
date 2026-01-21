import { useMemo } from 'react';
import { useGetTasks, useGetDayRecords } from '../hooks/useQueries';
import FailureHeatmap from './FailureHeatmap';
import WhatIfSimulator from './WhatIfSimulator';
import DataExport from './DataExport';
import { Download } from 'lucide-react';

export default function AnalyticsPanel() {
  const { data: tasks = [] } = useGetTasks();
  const { data: dayRecords = [] } = useGetDayRecords();

  const analytics = useMemo(() => {
    // Behavioral Pattern Detection
    const completedTasks = tasks.filter(t => t.status === 'completed');
    const failedTasks = tasks.filter(t => t.status === 'failed');
    
    const patterns: string[] = [];

    // Failure rate by category
    const categoryStats = new Map<string, { total: number; failed: number }>();
    tasks.forEach(task => {
      const cat = task.category || 'UNCATEGORIZED';
      const stats = categoryStats.get(cat) || { total: 0, failed: 0 };
      stats.total++;
      if (task.status === 'failed') stats.failed++;
      categoryStats.set(cat, stats);
    });

    categoryStats.forEach((stats, category) => {
      if (stats.total >= 3) {
        const failRate = (stats.failed / stats.total) * 100;
        if (failRate > 50) {
          patterns.push(`${category} tasks fail ${failRate.toFixed(0)}% of the time.`);
        }
      }
    });

    // Failure rate by difficulty
    const difficultyStats = new Map<number, { total: number; failed: number }>();
    tasks.forEach(task => {
      const diff = Number(task.difficulty);
      const stats = difficultyStats.get(diff) || { total: 0, failed: 0 };
      stats.total++;
      if (task.status === 'failed') stats.failed++;
      difficultyStats.set(diff, stats);
    });

    difficultyStats.forEach((stats, difficulty) => {
      if (stats.total >= 3) {
        const failRate = (stats.failed / stats.total) * 100;
        if (failRate > 60) {
          patterns.push(`Difficulty ${difficulty} tasks fail ${failRate.toFixed(0)}% of the time.`);
        }
      }
    });

    // Time-based patterns (simulated - would need timestamp data)
    if (failedTasks.length > 5) {
      patterns.push(`${failedTasks.length} tasks failed. Pattern analysis requires more data.`);
    }

    // Ambition vs Capacity Index
    let ambitionIndex = 0;
    let classification = 'INSUFFICIENT DATA';
    
    if (dayRecords.length >= 7) {
      const completionRatios = dayRecords
        .filter(r => r.plannedXP > 0)
        .map(r => r.completedXP / r.plannedXP);
      
      if (completionRatios.length > 0) {
        const avgCompletion = completionRatios.reduce((a, b) => a + b, 0) / completionRatios.length;
        const variance = completionRatios.reduce((sum, val) => sum + Math.pow(val - avgCompletion, 2), 0) / completionRatios.length;
        const stdDev = Math.sqrt(variance);
        
        ambitionIndex = avgCompletion;
        
        if (avgCompletion < 0.6) {
          classification = 'CHRONICALLY OVERCOMMITTING';
        } else if (avgCompletion > 0.9 && stdDev < 0.1) {
          classification = 'UNDER-AMBITIOUS';
        } else {
          classification = 'CALIBRATED';
        }
      }
    }

    // Collapse & Recovery Analysis
    const collapseEvents: { day: number; depth: number; recoveryDays: number }[] = [];
    let inCollapse = false;
    let collapseStart = 0;
    let collapseDepth = 0;

    dayRecords.forEach((record, index) => {
      const completionRate = record.plannedXP > 0 ? record.completedXP / record.plannedXP : 1;
      
      if (completionRate < 0.4 && !inCollapse) {
        inCollapse = true;
        collapseStart = index;
        collapseDepth = 1;
      } else if (inCollapse) {
        if (completionRate < 0.4) {
          collapseDepth++;
        } else if (completionRate > 0.7) {
          collapseEvents.push({
            day: collapseStart,
            depth: collapseDepth,
            recoveryDays: index - collapseStart - collapseDepth
          });
          inCollapse = false;
        }
      }
    });

    const avgRecoveryTime = collapseEvents.length > 0
      ? collapseEvents.reduce((sum, e) => sum + e.recoveryDays, 0) / collapseEvents.length
      : 0;

    const disciplineHalfLife = avgRecoveryTime > 0 ? avgRecoveryTime * 0.693 : 0;

    return {
      patterns,
      ambitionIndex,
      classification,
      collapseEvents,
      disciplineHalfLife,
      totalTasks: tasks.length,
      completedTasks: completedTasks.length,
      failedTasks: failedTasks.length
    };
  }, [tasks, dayRecords]);

  return (
    <div className="space-y-6">
      {/* Behavioral Patterns */}
      <div className="border border-[#8A8F98]/20 bg-[#0B0E11] p-6">
        <h2 className="mb-4 text-sm font-bold text-[#E6E6E6]">BEHAVIORAL PATTERN DETECTION</h2>
        <div className="space-y-2">
          {analytics.patterns.length === 0 ? (
            <div className="text-sm text-[#8A8F98]">INSUFFICIENT DATA FOR PATTERN ANALYSIS</div>
          ) : (
            analytics.patterns.map((pattern, index) => (
              <div key={index} className="border-l-2 border-[#DC5F5F] pl-3 text-sm text-[#E6E6E6]">
                {pattern}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Ambition vs Capacity */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="border border-[#8A8F98]/20 bg-[#0B0E11] p-6">
          <h2 className="mb-4 text-sm font-bold text-[#E6E6E6]">AMBITION VS CAPACITY INDEX</h2>
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="text-xs text-[#8A8F98]">COMPLETION RATIO</div>
              <div className="text-3xl font-bold text-[#E6E6E6]">
                {(analytics.ambitionIndex * 100).toFixed(1)}%
              </div>
            </div>
            <div className="space-y-2">
              <div className="text-xs text-[#8A8F98]">CLASSIFICATION</div>
              <div className={`text-lg font-bold ${
                analytics.classification === 'CALIBRATED' ? 'text-[#B6FF3B]' :
                analytics.classification === 'CHRONICALLY OVERCOMMITTING' ? 'text-[#DC5F5F]' :
                'text-[#8A8F98]'
              }`}>
                {analytics.classification}
              </div>
            </div>
          </div>
        </div>

        {/* Collapse & Recovery */}
        <div className="border border-[#8A8F98]/20 bg-[#0B0E11] p-6">
          <h2 className="mb-4 text-sm font-bold text-[#E6E6E6]">COLLAPSE & RECOVERY METRICS</h2>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="text-xs text-[#8A8F98]">COLLAPSE EVENTS</div>
                <div className="text-2xl font-bold text-[#DC5F5F]">
                  {analytics.collapseEvents.length}
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-xs text-[#8A8F98]">DISCIPLINE HALF-LIFE</div>
                <div className="text-2xl font-bold text-[#E6E6E6]">
                  {analytics.disciplineHalfLife.toFixed(1)} days
                </div>
              </div>
            </div>
            {analytics.collapseEvents.length > 0 && (
              <div className="space-y-1 border-t border-[#8A8F98]/20 pt-4">
                <div className="text-xs text-[#8A8F98]">WORST COLLAPSE</div>
                <div className="text-sm text-[#E6E6E6]">
                  {Math.max(...analytics.collapseEvents.map(e => e.depth))} consecutive days
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Failure Heatmap */}
      <FailureHeatmap />

      {/* What-If Simulator */}
      <WhatIfSimulator />

      {/* Data Export */}
      <DataExport />
    </div>
  );
}
