import { useGetSystemLogs } from '../hooks/useQueries';

export default function SystemLog() {
  const { data: logs = [] } = useGetSystemLogs();

  const sortedLogs = [...logs].sort((a, b) => Number(b.timestamp - a.timestamp));

  return (
    <div className="space-y-4 border border-[#8A8F98]/20 bg-[#0B0E11] p-6">
      <h2 className="text-sm font-bold text-[#E6E6E6]">SYSTEM INTEGRITY LOG</h2>
      
      <div className="space-y-2 font-mono text-xs">
        {sortedLogs.length === 0 ? (
          <div className="text-[#8A8F98]">NO EVENTS RECORDED</div>
        ) : (
          sortedLogs.map((log, index) => (
            <div key={index} className="border-l-2 border-[#8A8F98]/20 pl-4 py-2">
              <div className="flex items-center justify-between">
                <span className="text-[#8A8F98]">
                  DAY {log.day.toString()}
                </span>
                <span className="text-[#8A8F98]">
                  {new Date(Number(log.timestamp) / 1000000).toLocaleString()}
                </span>
              </div>
              <div className="mt-1 text-[#E6E6E6]">{log.eventType}</div>
              <div className="mt-1 text-[#8A8F98]">{log.details}</div>
              <div className={`mt-1 ${log.xpChange >= 0 ? 'text-[#B6FF3B]' : 'text-[#DC5F5F]'}`}>
                XP CHANGE: {log.xpChange >= 0 ? '+' : ''}{Math.floor(log.xpChange)}
              </div>
              {log.rankChange && (
                <div className="mt-1 text-[#DC5F5F]">RANK CHANGE: {log.rankChange}</div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
