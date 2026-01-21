import { useState } from 'react';
import Header from '../components/Header';
import TaskList from '../components/TaskList';
import XPOverview from '../components/XPOverview';
import PerformanceGraph from '../components/PerformanceGraph';
import SystemLog from '../components/SystemLog';
import LockInPanel from '../components/LockInPanel';
import AnalyticsPanel from '../components/AnalyticsPanel';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="min-h-screen bg-[#0B0E11]">
      <Header />
      
      <main className="mx-auto max-w-7xl px-4 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="border-b border-[#8A8F98]/20 bg-transparent">
            <TabsTrigger 
              value="overview" 
              className="data-[state=active]:border-b-2 data-[state=active]:border-[#B6FF3B] data-[state=active]:text-[#B6FF3B]"
            >
              OVERVIEW
            </TabsTrigger>
            <TabsTrigger 
              value="tasks" 
              className="data-[state=active]:border-b-2 data-[state=active]:border-[#B6FF3B] data-[state=active]:text-[#B6FF3B]"
            >
              TASKS
            </TabsTrigger>
            <TabsTrigger 
              value="analytics" 
              className="data-[state=active]:border-b-2 data-[state=active]:border-[#B6FF3B] data-[state=active]:text-[#B6FF3B]"
            >
              ANALYTICS
            </TabsTrigger>
            <TabsTrigger 
              value="performance" 
              className="data-[state=active]:border-b-2 data-[state=active]:border-[#B6FF3B] data-[state=active]:text-[#B6FF3B]"
            >
              PERFORMANCE
            </TabsTrigger>
            <TabsTrigger 
              value="log" 
              className="data-[state=active]:border-b-2 data-[state=active]:border-[#B6FF3B] data-[state=active]:text-[#B6FF3B]"
            >
              SYSTEM LOG
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <XPOverview />
              </div>
              <div>
                <LockInPanel />
              </div>
            </div>
            <PerformanceGraph />
          </TabsContent>

          <TabsContent value="tasks">
            <TaskList />
          </TabsContent>

          <TabsContent value="analytics">
            <AnalyticsPanel />
          </TabsContent>

          <TabsContent value="performance">
            <PerformanceGraph fullView />
          </TabsContent>

          <TabsContent value="log">
            <SystemLog />
          </TabsContent>
        </Tabs>
      </main>

      <footer className="border-t border-[#8A8F98]/20 py-6 text-center text-xs text-[#8A8F98]">
        © 2025. Built with Discipline not motivation.
      </footer>
    </div>
  );
}
