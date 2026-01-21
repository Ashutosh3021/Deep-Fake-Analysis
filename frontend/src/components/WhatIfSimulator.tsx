import { useState, useMemo } from 'react';
import { useGetTasks, useGetDayRecords, useGetXPState } from '../hooks/useQueries';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

export default function WhatIfSimulator() {
  const { data: tasks = [] } = useGetTasks();
  const { data: dayRecords = [] } = useGetDayRecords();
  const { data: xpState } = useGetXPState();

  const [simulationMode, setSimulationMode] = useState<'adjust' | 'remove' | 'add'>('adjust');
  const [selectedTaskId, setSelectedTaskId] = useState<string>('');
  const [xpAdjustment, setXpAdjustment] = useState<number>(0);

  const pendingTasks = tasks.filter(t => t.status === 'pending');

  const projection = useMemo(() => {
    if (!xpState) return null;

    let projectedXP = xpState.totalXP;
    let projectedPlannedXP = 0;
    let projectedCompletedXP = 0;

    pendingTasks.forEach(task => {
      let taskXP = task.plannedXP;
      
      if (simulationMode === 'adjust' && task.id === selectedTaskId) {
        taskXP += xpAdjustment;
      } else if (simulationMode === 'remove' && task.id === selectedTaskId) {
        taskXP = 0;
      }

      projectedPlannedXP += taskXP;
      // Assume 70% completion for projection
      projectedCompletedXP += taskXP * 0.7;
    });

    if (simulationMode === 'add') {
      projectedPlannedXP += xpAdjustment;
      projectedCompletedXP += xpAdjustment * 0.7;
    }

    projectedXP += projectedCompletedXP;

    // Calculate projected rank
    const rankThresholds = [
      { name: 'NOVICE', floor: 0, ceiling: 1000 },
      { name: 'INTERMEDIATE', floor: 1000, ceiling: 2500 },
      { name: 'ADVANCED', floor: 2500, ceiling: 5000 },
      { name: 'EXPERT', floor: 5000, ceiling: 10000 },
      { name: 'MASTER', floor: 10000, ceiling: Infinity },
    ];

    const projectedRank = rankThresholds.find(r => projectedXP >= r.floor && projectedXP < r.ceiling)?.name || 'MASTER';

    return {
      currentXP: xpState.totalXP,
      projectedXP,
      projectedPlannedXP,
      projectedCompletedXP,
      projectedRank,
      xpChange: projectedXP - xpState.totalXP
    };
  }, [xpState, pendingTasks, simulationMode, selectedTaskId, xpAdjustment]);

  return (
    <div className="border border-[#8A8F98]/20 bg-[#0B0E11] p-6">
      <h2 className="mb-4 text-sm font-bold text-[#E6E6E6]">WHAT-IF SIMULATOR</h2>
      
      <div className="space-y-4">
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="space-y-2">
            <label className="text-xs text-[#8A8F98]">SIMULATION MODE</label>
            <Select value={simulationMode} onValueChange={(v: any) => setSimulationMode(v)}>
              <SelectTrigger className="border-[#8A8F98]/20 bg-transparent text-[#E6E6E6]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="border-[#8A8F98]/20 bg-[#0B0E11]">
                <SelectItem value="adjust">ADJUST TASK XP</SelectItem>
                <SelectItem value="remove">REMOVE TASK</SelectItem>
                <SelectItem value="add">ADD NEW TASK</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {simulationMode !== 'add' && (
            <div className="space-y-2">
              <label className="text-xs text-[#8A8F98]">SELECT TASK</label>
              <Select value={selectedTaskId} onValueChange={setSelectedTaskId}>
                <SelectTrigger className="border-[#8A8F98]/20 bg-transparent text-[#E6E6E6]">
                  <SelectValue placeholder="Choose task..." />
                </SelectTrigger>
                <SelectContent className="border-[#8A8F98]/20 bg-[#0B0E11]">
                  {pendingTasks.map(task => (
                    <SelectItem key={task.id} value={task.id}>
                      {task.title} ({task.plannedXP} XP)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {(simulationMode === 'adjust' || simulationMode === 'add') && (
            <div className="space-y-2">
              <label className="text-xs text-[#8A8F98]">
                {simulationMode === 'add' ? 'NEW TASK XP' : 'XP ADJUSTMENT'}
              </label>
              <Input
                type="number"
                value={xpAdjustment}
                onChange={(e) => setXpAdjustment(parseFloat(e.target.value) || 0)}
                className="border-[#8A8F98]/20 bg-transparent text-[#E6E6E6]"
                placeholder={simulationMode === 'add' ? 'Enter XP value' : '+/- XP'}
              />
            </div>
          )}
        </div>

        {projection && (
          <div className="border-t border-[#8A8F98]/20 pt-4">
            <div className="grid gap-4 lg:grid-cols-4">
              <div className="space-y-1">
                <div className="text-xs text-[#8A8F98]">CURRENT XP</div>
                <div className="text-xl font-bold text-[#E6E6E6]">
                  {Math.floor(projection.currentXP)}
                </div>
              </div>
              <div className="space-y-1">
                <div className="text-xs text-[#8A8F98]">PROJECTED XP</div>
                <div className="text-xl font-bold text-[#B6FF3B]">
                  {Math.floor(projection.projectedXP)}
                </div>
              </div>
              <div className="space-y-1">
                <div className="text-xs text-[#8A8F98]">XP CHANGE</div>
                <div className={`text-xl font-bold ${projection.xpChange >= 0 ? 'text-[#B6FF3B]' : 'text-[#DC5F5F]'}`}>
                  {projection.xpChange >= 0 ? '+' : ''}{Math.floor(projection.xpChange)}
                </div>
              </div>
              <div className="space-y-1">
                <div className="text-xs text-[#8A8F98]">PROJECTED RANK</div>
                <div className="text-xl font-bold text-[#E6E6E6]">
                  {projection.projectedRank}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
