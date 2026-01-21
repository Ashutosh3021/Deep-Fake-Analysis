import { Button } from '@/components/ui/button';
import { useGetTasks, useGetDayRecords, useGetSystemLogs, useGetXPState, useGetRank } from '../hooks/useQueries';
import { Download } from 'lucide-react';

export default function DataExport() {
  const { data: tasks = [] } = useGetTasks();
  const { data: dayRecords = [] } = useGetDayRecords();
  const { data: systemLogs = [] } = useGetSystemLogs();
  const { data: xpState } = useGetXPState();
  const { data: rank } = useGetRank();

  const exportJSON = () => {
    const data = {
      exportDate: new Date().toISOString(),
      xpState,
      rank,
      tasks: tasks.map(t => ({
        ...t,
        owner: t.owner.toString(),
        dueDate: t.dueDate.toString(),
        difficulty: t.difficulty.toString()
      })),
      dayRecords: dayRecords.map(r => ({
        ...r,
        owner: r.owner.toString(),
        date: r.date.toString()
      })),
      systemLogs: systemLogs.map(l => ({
        ...l,
        owner: l.owner.toString(),
        timestamp: l.timestamp.toString(),
        day: l.day.toString()
      }))
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lifequest-export-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportCSV = () => {
    // Export day records as CSV
    const headers = ['Date', 'Planned XP', 'Completed XP', 'Earned XP', 'Penalties', 'XP Debt'];
    const rows = dayRecords.map(r => [
      new Date(Number(r.date) / 1000000).toISOString(),
      r.plannedXP,
      r.completedXP,
      r.earnedXP,
      r.penaltiesApplied,
      r.XPdebt
    ]);

    const csv = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lifequest-dayrecords-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="border border-[#8A8F98]/20 bg-[#0B0E11] p-6">
      <h2 className="mb-4 text-sm font-bold text-[#E6E6E6]">DATA EXPORT</h2>
      <div className="flex gap-4">
        <Button
          onClick={exportJSON}
          className="border border-[#B6FF3B] bg-transparent text-[#B6FF3B] hover:bg-[#B6FF3B]/10"
        >
          <Download className="mr-2 h-4 w-4" />
          EXPORT JSON
        </Button>
        <Button
          onClick={exportCSV}
          className="border border-[#B6FF3B] bg-transparent text-[#B6FF3B] hover:bg-[#B6FF3B]/10"
        >
          <Download className="mr-2 h-4 w-4" />
          EXPORT CSV
        </Button>
      </div>
      <div className="mt-4 text-xs text-[#8A8F98]">
        Export includes: XP logs, daily records, rank transitions, and performance data.
      </div>
    </div>
  );
}
