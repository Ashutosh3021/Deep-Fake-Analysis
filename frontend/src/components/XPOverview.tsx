import { useGetXPState, useGetRank, useGetDayRecords } from '../hooks/useQueries';

export default function XPOverview() {
  const { data: xpState } = useGetXPState();
  const { data: rank } = useGetRank();
  const { data: dayRecords } = useGetDayRecords();

  const currentDay = dayRecords?.length || 0;
  const isWarmUp = currentDay <= 7;

  const bufferPercentage = rank ? (rank.surplusBuffer / 500) * 100 : 0;
  const rankProgress = rank ? ((xpState?.totalXP || 0) - rank.xpFloor) / (rank.xpCeiling - rank.xpFloor) * 100 : 0;

  // Calculate ambition vs capacity indicator
  const recentRecords = dayRecords?.slice(-7) || [];
  const avgCompletion = recentRecords.length > 0
    ? recentRecords.reduce((sum, r) => sum + (r.plannedXP > 0 ? r.completedXP / r.plannedXP : 0), 0) / recentRecords.length
    : 0;

  return (
    <div className="space-y-4 border border-[#8A8F98]/20 bg-[#0B0E11] p-6">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold text-[#E6E6E6]">XP STATE</h2>
        {isWarmUp && (
          <span className="text-xs text-[#B6FF3B]">WARM-UP BUFFER ACTIVE (DAY {currentDay}/7)</span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1">
          <div className="text-xs text-[#8A8F98]">TOTAL XP</div>
          <div className="text-2xl font-bold text-[#E6E6E6]">{Math.floor(xpState?.totalXP || 0)}</div>
        </div>

        <div className="space-y-1">
          <div className="text-xs text-[#8A8F98]">XP DEBT</div>
          <div className="text-2xl font-bold text-[#DC5F5F]">{Math.floor(xpState?.debt || 0)}</div>
        </div>

        <div className="space-y-1">
          <div className="text-xs text-[#8A8F98]">DAILY EARNED</div>
          <div className="text-xl font-bold text-[#B6FF3B]">+{Math.floor(xpState?.dailyEarnedXP || 0)}</div>
        </div>

        <div className="space-y-1">
          <div className="text-xs text-[#8A8F98]">DAILY LOST</div>
          <div className="text-xl font-bold text-[#DC5F5F]">-{Math.floor(xpState?.dailyLostXP || 0)}</div>
        </div>
      </div>

      <div className="space-y-2 border-t border-[#8A8F98]/20 pt-4">
        <div className="flex items-center justify-between text-xs">
          <span className="text-[#8A8F98]">RANK PROGRESS</span>
          <span className="text-[#E6E6E6]">{rank?.currentRank || 'NOVICE'}</span>
        </div>
        <div className="h-2 bg-[#8A8F98]/20">
          <div 
            className="h-full bg-[#B6FF3B]" 
            style={{ width: `${Math.min(100, Math.max(0, rankProgress))}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-[#8A8F98]">
          <span>{Math.floor(rank?.xpFloor || 0)}</span>
          <span>{Math.floor(rank?.xpCeiling || 0)}</span>
        </div>
      </div>

      <div className="space-y-2 border-t border-[#8A8F98]/20 pt-4">
        <div className="flex items-center justify-between text-xs">
          <span className="text-[#8A8F98]">SURPLUS BUFFER</span>
          <span className={bufferPercentage < 30 ? 'text-[#DC5F5F]' : 'text-[#E6E6E6]'}>
            {Math.floor(rank?.surplusBuffer || 0)} / 500
          </span>
        </div>
        <div className="h-2 bg-[#8A8F98]/20">
          <div 
            className={`h-full ${bufferPercentage < 30 ? 'bg-[#DC5F5F]' : 'bg-[#8A8F98]'}`}
            style={{ width: `${bufferPercentage}%` }}
          />
        </div>
        {rank?.demotionFlag && (
          <div className="text-xs text-[#DC5F5F]">DEMOTION IMMINENT</div>
        )}
      </div>

      {recentRecords.length >= 3 && (
        <div className="border-t border-[#8A8F98]/20 pt-4">
          <div className="flex items-center justify-between text-xs">
            <span className="text-[#8A8F98]">CAPACITY INDEX (7-DAY)</span>
            <span className={`${
              avgCompletion < 0.6 ? 'text-[#DC5F5F]' :
              avgCompletion > 0.9 ? 'text-[#8A8F98]' :
              'text-[#B6FF3B]'
            }`}>
              {(avgCompletion * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
