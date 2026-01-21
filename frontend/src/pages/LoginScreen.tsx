import { useInternetIdentity } from '../hooks/useInternetIdentity';
import { Button } from '@/components/ui/button';

export default function LoginScreen() {
  const { login, loginStatus } = useInternetIdentity();

  const isLoggingIn = loginStatus === 'logging-in';

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-[#0B0E11] px-4">
      <div className="w-full max-w-md space-y-8">
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tight text-[#E6E6E6]">LIFEQUEST</h1>
          <p className="text-sm text-[#8A8F98]">ANALYTICAL TASK MANAGEMENT SYSTEM</p>
        </div>

        <div className="space-y-4 border border-[#8A8F98]/20 bg-[#0B0E11] p-6">
          <div className="space-y-2">
            <p className="text-xs text-[#8A8F98]">SYSTEM REQUIREMENTS</p>
            <ul className="space-y-1 text-xs text-[#8A8F98]">
              <li>• Authentication required</li>
              <li>• Daily XP tracking enabled</li>
              <li>• Penalty system active</li>
              <li>• No grace periods</li>
            </ul>
          </div>

          <Button
            onClick={login}
            disabled={isLoggingIn}
            className="w-full border border-[#B6FF3B] bg-transparent text-[#B6FF3B] hover:bg-[#B6FF3B]/10"
          >
            {isLoggingIn ? 'AUTHENTICATING...' : 'AUTHENTICATE'}
          </Button>
        </div>

        <div className="text-xs text-[#8A8F98]">
          <p>This system tracks performance without forgiveness.</p>
          <p className="mt-1">Proceed only if you accept full accountability.</p>
        </div>
      </div>
    </div>
  );
}
