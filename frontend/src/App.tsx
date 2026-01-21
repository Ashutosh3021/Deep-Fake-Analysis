import { useInternetIdentity } from './hooks/useInternetIdentity';
import { useGetCallerUserProfile } from './hooks/useQueries';
import LoginScreen from './pages/LoginScreen';
import ProfileSetup from './pages/ProfileSetup';
import Dashboard from './pages/Dashboard';

export default function App() {
  const { identity } = useInternetIdentity();
  const { data: userProfile, isLoading: profileLoading, isFetched } = useGetCallerUserProfile();

  const isAuthenticated = !!identity;

  if (!isAuthenticated) {
    return <LoginScreen />;
  }

  if (profileLoading || !isFetched) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#0B0E11]">
        <div className="text-[#8A8F98]">INITIALIZING SYSTEM</div>
      </div>
    );
  }

  if (userProfile === null) {
    return <ProfileSetup />;
  }

  return <Dashboard />;
}
