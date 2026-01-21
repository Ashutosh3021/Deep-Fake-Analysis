import { useMemo } from 'react';
import { useGetTasks } from '../hooks/useQueries';

export default function FailureHeatmap() {
  const { data: tasks = [] } = useGetTasks();

  const heatmapData = useMemo(() => {
    // Group by category and day of week
    const categories = new Set<string>();
    const dayNames = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'];
    
    tasks.forEach(task => {
      categories.add(task.category || 'UNCATEGORIZED');
    });

    const categoryArray = Array.from(categories).sort();
    const grid: number[][] = [];

    categoryArray.forEach(category => {
      const row: number[] = [];
      dayNames.forEach((_, dayIndex) => {
        const categoryTasks = tasks.filter(t => {
          const taskDay = new Date(Number(t.dueDate) / 1000000).getDay();
          return (t.category || 'UNCATEGORIZED') === category && taskDay === dayIndex;
        });
        
        const failedCount = categoryTasks.filter(t => t.status === 'failed').length;
        const failRate = categoryTasks.length > 0 ? failedCount / categoryTasks.length : 0;
        row.push(failRate);
      });
      grid.push(row);
    });

    return { categories: categoryArray, dayNames, grid };
  }, [tasks]);

  const getHeatColor = (value: number) => {
    if (value === 0) return 'bg-[#8A8F98]/10';
    if (value < 0.3) return 'bg-[#8A8F98]/30';
    if (value < 0.5) return 'bg-[#DC5F5F]/30';
    if (value < 0.7) return 'bg-[#DC5F5F]/60';
    return 'bg-[#DC5F5F]';
  };

  if (heatmapData.categories.length === 0) {
    return (
      <div className="border border-[#8A8F98]/20 bg-[#0B0E11] p-6">
        <h2 className="mb-4 text-sm font-bold text-[#E6E6E6]">FAILURE CLUSTER HEATMAP</h2>
        <div className="text-sm text-[#8A8F98]">INSUFFICIENT DATA FOR HEATMAP GENERATION</div>
      </div>
    );
  }

  return (
    <div className="border border-[#8A8F98]/20 bg-[#0B0E11] p-6">
      <h2 className="mb-4 text-sm font-bold text-[#E6E6E6]">FAILURE CLUSTER HEATMAP</h2>
      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          <div className="flex">
            <div className="w-32 flex-shrink-0" />
            <div className="flex flex-1 gap-1">
              {heatmapData.dayNames.map(day => (
                <div key={day} className="flex-1 text-center text-xs text-[#8A8F98]">
                  {day}
                </div>
              ))}
            </div>
          </div>
          <div className="mt-2 space-y-1">
            {heatmapData.categories.map((category, catIndex) => (
              <div key={category} className="flex items-center">
                <div className="w-32 flex-shrink-0 truncate pr-2 text-xs text-[#8A8F98]">
                  {category}
                </div>
                <div className="flex flex-1 gap-1">
                  {heatmapData.grid[catIndex].map((value, dayIndex) => (
                    <div
                      key={dayIndex}
                      className={`flex-1 ${getHeatColor(value)}`}
                      style={{ height: '24px' }}
                      title={`${category} - ${heatmapData.dayNames[dayIndex]}: ${(value * 100).toFixed(0)}% failure rate`}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 flex items-center justify-end gap-2 text-xs text-[#8A8F98]">
            <span>0%</span>
            <div className="flex gap-1">
              <div className="h-4 w-4 bg-[#8A8F98]/10" />
              <div className="h-4 w-4 bg-[#8A8F98]/30" />
              <div className="h-4 w-4 bg-[#DC5F5F]/30" />
              <div className="h-4 w-4 bg-[#DC5F5F]/60" />
              <div className="h-4 w-4 bg-[#DC5F5F]" />
            </div>
            <span>100%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
