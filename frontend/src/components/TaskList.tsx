import { useState, useMemo } from 'react';
import { useGetTasks, useCreateTask, useUpdateTask, useDeleteTask, useGetDayRecords } from '../hooks/useQueries';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Trash2, Plus, AlertTriangle } from 'lucide-react';
import type { Task, TaskStatus } from '../backend';

export default function TaskList() {
  const { data: tasks = [] } = useGetTasks();
  const { data: dayRecords = [] } = useGetDayRecords();
  const createTask = useCreateTask();
  const updateTask = useUpdateTask();
  const deleteTask = useDeleteTask();

  const [isCreateOpen, setIsCreateOpen] = useState(false);

  const [formData, setFormData] = useState({
    title: '',
    description: '',
    category: '',
    difficulty: 1,
    plannedXP: 0,
    dueDate: new Date().toISOString().split('T')[0],
    isCritical: false,
  });

  const resetForm = () => {
    setFormData({
      title: '',
      description: '',
      category: '',
      difficulty: 1,
      plannedXP: 0,
      dueDate: new Date().toISOString().split('T')[0],
      isCritical: false,
    });
  };

  // Calculate failure probability for each task
  const taskFailureProbabilities = useMemo(() => {
    const probabilities = new Map<string, number>();

    tasks.forEach(task => {
      if (task.status !== 'pending') return;

      // Calculate failure probability based on historical data
      const similarTasks = tasks.filter(t => 
        t.category === task.category && 
        t.difficulty === task.difficulty &&
        t.status !== 'pending'
      );

      if (similarTasks.length >= 3) {
        const failedCount = similarTasks.filter(t => t.status === 'failed').length;
        const failRate = (failedCount / similarTasks.length) * 100;
        probabilities.set(task.id, Math.round(failRate));
      } else {
        // Default probability based on difficulty
        const baseProbability = Number(task.difficulty) * 8;
        probabilities.set(task.id, Math.min(100, baseProbability));
      }
    });

    return probabilities;
  }, [tasks]);

  const handleCreate = () => {
    createTask.mutate({
      id: Date.now().toString(),
      title: formData.title,
      description: formData.description,
      category: formData.category || undefined,
      difficulty: BigInt(formData.difficulty),
      plannedXP: formData.plannedXP,
      dueDate: BigInt(new Date(formData.dueDate).getTime() * 1000000),
      recurrence: undefined,
      isCritical: formData.isCritical,
    });
    setIsCreateOpen(false);
    resetForm();
  };

  const handleUpdate = (task: Task, updates: Partial<Task>) => {
    // Prevent modification of critical tasks
    if (task.isCritical && (updates.plannedXP !== undefined || updates.difficulty !== undefined)) {
      alert('CRITICAL TASK: Cannot modify XP or difficulty of locked tasks.');
      return;
    }
    updateTask.mutate({ ...task, ...updates });
  };

  const handleDelete = (task: Task) => {
    if (task.isCritical) {
      alert('CRITICAL TASK: Cannot delete locked tasks.');
      return;
    }
    if (confirm('Delete this task? This action cannot be undone.')) {
      deleteTask.mutate(task.id);
    }
  };

  const sortedTasks = [...tasks].sort((a, b) => {
    if (a.status === 'pending' && b.status !== 'pending') return -1;
    if (a.status !== 'pending' && b.status === 'pending') return 1;
    return Number(a.dueDate - b.dueDate);
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold text-[#E6E6E6]">TASK REGISTRY</h2>
        <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
          <DialogTrigger asChild>
            <Button className="border border-[#B6FF3B] bg-transparent text-[#B6FF3B] hover:bg-[#B6FF3B]/10">
              <Plus className="mr-2 h-4 w-4" />
              NEW TASK
            </Button>
          </DialogTrigger>
          <DialogContent className="border-[#8A8F98]/20 bg-[#0B0E11]">
            <DialogHeader>
              <DialogTitle className="text-[#E6E6E6]">CREATE TASK</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-xs text-[#8A8F98]">TITLE</label>
                <Input
                  value={formData.title}
                  onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                  className="border-[#8A8F98]/20 bg-transparent text-[#E6E6E6]"
                />
              </div>
              <div className="space-y-2">
                <label className="text-xs text-[#8A8F98]">DESCRIPTION</label>
                <Textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="border-[#8A8F98]/20 bg-transparent text-[#E6E6E6]"
                />
              </div>
              <div className="space-y-2">
                <label className="text-xs text-[#8A8F98]">CATEGORY</label>
                <Input
                  value={formData.category}
                  onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                  className="border-[#8A8F98]/20 bg-transparent text-[#E6E6E6]"
                  placeholder="Optional"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-xs text-[#8A8F98]">DIFFICULTY</label>
                  <Input
                    type="number"
                    min="1"
                    max="10"
                    value={formData.difficulty}
                    onChange={(e) => setFormData({ ...formData, difficulty: parseInt(e.target.value) })}
                    className="border-[#8A8F98]/20 bg-transparent text-[#E6E6E6]"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs text-[#8A8F98]">PLANNED XP</label>
                  <Input
                    type="number"
                    min="0"
                    value={formData.plannedXP}
                    onChange={(e) => setFormData({ ...formData, plannedXP: parseFloat(e.target.value) })}
                    className="border-[#8A8F98]/20 bg-transparent text-[#E6E6E6]"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-xs text-[#8A8F98]">DUE DATE</label>
                <Input
                  type="date"
                  value={formData.dueDate}
                  onChange={(e) => setFormData({ ...formData, dueDate: e.target.value })}
                  className="border-[#8A8F98]/20 bg-transparent text-[#E6E6E6]"
                />
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="critical"
                  checked={formData.isCritical}
                  onCheckedChange={(checked) => setFormData({ ...formData, isCritical: checked as boolean })}
                />
                <label htmlFor="critical" className="text-xs text-[#8A8F98]">
                  MARK AS CRITICAL (locks XP/difficulty, prevents deletion)
                </label>
              </div>
              <Button
                onClick={handleCreate}
                disabled={!formData.title || createTask.isPending}
                className="w-full border border-[#B6FF3B] bg-transparent text-[#B6FF3B] hover:bg-[#B6FF3B]/10"
              >
                {createTask.isPending ? 'CREATING...' : 'CREATE'}
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <div className="space-y-2">
        {sortedTasks.length === 0 ? (
          <div className="border border-[#8A8F98]/20 bg-[#0B0E11] p-8 text-center text-[#8A8F98]">
            NO TASKS REGISTERED
          </div>
        ) : (
          sortedTasks.map((task) => {
            const failureProbability = taskFailureProbabilities.get(task.id);
            
            return (
              <div
                key={task.id}
                className="border border-[#8A8F98]/20 bg-[#0B0E11] p-4"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-3">
                      <h3 className="font-bold text-[#E6E6E6]">{task.title}</h3>
                      {task.isCritical && (
                        <span className="flex items-center gap-1 text-xs text-[#DC5F5F]">
                          <AlertTriangle className="h-3 w-3" />
                          CRITICAL
                        </span>
                      )}
                      <span className={`text-xs ${
                        task.status === 'completed' ? 'text-[#B6FF3B]' :
                        task.status === 'failed' ? 'text-[#DC5F5F]' :
                        'text-[#8A8F98]'
                      }`}>
                        {task.status.toUpperCase()}
                      </span>
                      {task.status === 'pending' && failureProbability !== undefined && (
                        <span className={`text-xs ${
                          failureProbability > 70 ? 'text-[#DC5F5F]' :
                          failureProbability > 40 ? 'text-[#8A8F98]' :
                          'text-[#B6FF3B]'
                        }`}>
                          FAIL PROBABILITY: {failureProbability}%
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-[#8A8F98]">{task.description}</p>
                    <div className="flex gap-4 text-xs text-[#8A8F98]">
                      {task.category && <span>CATEGORY: {task.category}</span>}
                      <span>DIFFICULTY: {task.difficulty.toString()}</span>
                      <span>XP: {task.plannedXP}</span>
                      <span>DUE: {new Date(Number(task.dueDate) / 1000000).toLocaleDateString()}</span>
                    </div>
                    {task.status === 'pending' && (
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          onClick={() => handleUpdate(task, { status: 'completed' as TaskStatus, completionFactor: 1 })}
                          className="border border-[#B6FF3B] bg-transparent text-[#B6FF3B] hover:bg-[#B6FF3B]/10"
                        >
                          COMPLETE
                        </Button>
                        <Button
                          size="sm"
                          onClick={() => handleUpdate(task, { status: 'failed' as TaskStatus })}
                          className="border border-[#DC5F5F] bg-transparent text-[#DC5F5F] hover:bg-[#DC5F5F]/10"
                        >
                          FAIL
                        </Button>
                      </div>
                    )}
                  </div>
                  {!task.isCritical && (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleDelete(task)}
                      className="text-[#8A8F98] hover:text-[#DC5F5F]"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
