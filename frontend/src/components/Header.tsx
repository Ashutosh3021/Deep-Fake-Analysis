import { useInternetIdentity } from '../hooks/useInternetIdentity';
import { useGetCallerUserProfile, useGetXPState, useGetRank } from '../hooks/useQueries';
import { useQueryClient } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';

export default function Header() {
  const { clear } = useInternetIdentity();
  const queryClient = useQueryClient();
  const { data: profile } = useGetCallerUserProfile();
  const { data: xpState } = useGetXPState();
  const { data: rank } = useGetRank();

  const handleLogout = async () => {
    await clear();
    queryClient.clear();
  };

  return (
    <header className="border-b border-[#8A8F98]/20 bg-[#0B0E11]">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4">
        <div className="space-y-1">
          <h1 className="text-xl font-bold tracking-tight text-[#E6E6E6]">LIFEQUEST</h1>
          <p className="text-xs text-[#8A8F98]">{profile?.name || 'USER'}</p>
        </div>

        <div className="flex items-center gap-6">
          <div className="space-y-1 text-right">
            <div className="text-xs text-[#8A8F98]">RANK</div>
            <div className="text-sm font-bold text-[#E6E6E6]">{rank?.currentRank || 'NOVICE'}</div>
          </div>

          <div className="space-y-1 text-right">
            <div className="text-xs text-[#8A8F98]">TOTAL XP</div>
            <div className="text-sm font-bold text-[#E6E6E6]">{Math.floor(xpState?.totalXP || 0)}</div>
          </div>

          <Button
            onClick={handleLogout}
            variant="ghost"
            className="border border-[#8A8F98]/20 text-[#8A8F98] hover:bg-[#8A8F98]/10 hover:text-[#E6E6E6]"
          >
            LOGOUT
          </Button>
        </div>
      </div>
    </header>
  );
}
