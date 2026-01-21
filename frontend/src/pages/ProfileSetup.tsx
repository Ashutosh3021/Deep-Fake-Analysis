import { useState } from 'react';
import { useSaveCallerUserProfile } from '../hooks/useQueries';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

export default function ProfileSetup() {
  const [name, setName] = useState('');
  const saveProfile = useSaveCallerUserProfile();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (name.trim()) {
      saveProfile.mutate({
        name: name.trim(),
        createdAt: BigInt(Date.now() * 1000000),
      });
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-[#0B0E11] px-4">
      <div className="w-full max-w-md space-y-8">
        <div className="space-y-2">
          <h1 className="text-2xl font-bold tracking-tight text-[#E6E6E6]">IDENTITY REQUIRED</h1>
          <p className="text-sm text-[#8A8F98]">Enter your identifier for system records</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <label className="text-xs text-[#8A8F98]">NAME</label>
            <Input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter name"
              className="border-[#8A8F98]/20 bg-transparent text-[#E6E6E6] placeholder:text-[#8A8F98]/50"
              required
            />
          </div>

          <Button
            type="submit"
            disabled={!name.trim() || saveProfile.isPending}
            className="w-full border border-[#B6FF3B] bg-transparent text-[#B6FF3B] hover:bg-[#B6FF3B]/10"
          >
            {saveProfile.isPending ? 'INITIALIZING...' : 'INITIALIZE SYSTEM'}
          </Button>
        </form>
      </div>
    </div>
  );
}
