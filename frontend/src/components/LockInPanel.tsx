import { useState } from 'react';
import { useGetLockInMode, useUpdateLockInMode } from '../hooks/useQueries';
import { Button } from '@/components/ui/button';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';

export default function LockInPanel() {
  const { data: lockInMode } = useGetLockInMode();
  const updateLockIn = useUpdateLockInMode();
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);

  const handleActivate = () => {
    updateLockIn.mutate({
      isActive: true,
      startDate: BigInt(Date.now() * 1000000),
      daysRemaining: BigInt(30),
    });
    setIsConfirmOpen(false);
  };

  if (lockInMode?.isActive) {
    return (
      <div className="space-y-4 border border-[#DC5F5F] bg-[#0B0E11] p-6">
        <h2 className="text-sm font-bold text-[#DC5F5F]">LOCK-IN MODE ACTIVE</h2>
        <div className="space-y-2">
          <div className="text-xs text-[#8A8F98]">DAYS REMAINING</div>
          <div className="text-3xl font-bold text-[#E6E6E6]">{lockInMode.daysRemaining.toString()}</div>
        </div>
        <div className="text-xs text-[#8A8F98]">
          All XP and rank rules are frozen. This mode cannot be disabled.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4 border border-[#8A8F98]/20 bg-[#0B0E11] p-6">
      <h2 className="text-sm font-bold text-[#E6E6E6]">LOCK-IN MODE</h2>
      <div className="space-y-2 text-xs text-[#8A8F98]">
        <p>30-day irreversible commitment mode</p>
        <p>• Freezes all XP calculation rules</p>
        <p>• Cannot be disabled once activated</p>
        <p>• No modifications to rank system</p>
      </div>

      <AlertDialog open={isConfirmOpen} onOpenChange={setIsConfirmOpen}>
        <AlertDialogTrigger asChild>
          <Button className="w-full border border-[#DC5F5F] bg-transparent text-[#DC5F5F] hover:bg-[#DC5F5F]/10">
            ACTIVATE LOCK-IN
          </Button>
        </AlertDialogTrigger>
        <AlertDialogContent className="border-[#8A8F98]/20 bg-[#0B0E11]">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-[#DC5F5F]">IRREVERSIBLE ACTION</AlertDialogTitle>
            <AlertDialogDescription className="text-[#8A8F98]">
              This will activate a 30-day lock-in period. All XP and rank calculation rules will be frozen.
              This action cannot be undone or disabled. Proceed only if you accept full commitment.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="border-[#8A8F98]/20 bg-transparent text-[#8A8F98]">
              CANCEL
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleActivate}
              className="border border-[#DC5F5F] bg-transparent text-[#DC5F5F] hover:bg-[#DC5F5F]/10"
            >
              CONFIRM LOCK-IN
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
