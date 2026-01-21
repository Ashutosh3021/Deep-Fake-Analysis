import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useActor } from './useActor';
import type { UserProfile, Task, CreateTaskData, DayRecord, SystemLogEntry, XPState, LockInMode, Rank } from '../backend';

// User Profile Queries
export function useGetCallerUserProfile() {
  const { actor, isFetching: actorFetching } = useActor();

  const query = useQuery<UserProfile | null>({
    queryKey: ['currentUserProfile'],
    queryFn: async () => {
      if (!actor) throw new Error('Actor not available');
      return actor.getCallerUserProfile();
    },
    enabled: !!actor && !actorFetching,
    retry: false,
  });

  return {
    ...query,
    isLoading: actorFetching || query.isLoading,
    isFetched: !!actor && query.isFetched,
  };
}

export function useSaveCallerUserProfile() {
  const { actor } = useActor();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (profile: UserProfile) => {
      if (!actor) throw new Error('Actor not available');
      await actor.saveCallerUserProfile(profile);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['currentUserProfile'] });
    },
  });
}

// Task Queries
export function useGetTasks() {
  const { actor, isFetching } = useActor();

  return useQuery<Task[]>({
    queryKey: ['tasks'],
    queryFn: async () => {
      if (!actor) return [];
      return actor.getTasks();
    },
    enabled: !!actor && !isFetching,
  });
}

export function useCreateTask() {
  const { actor } = useActor();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (taskData: CreateTaskData) => {
      if (!actor) throw new Error('Actor not available');
      await actor.createTask(taskData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });
}

export function useUpdateTask() {
  const { actor } = useActor();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (task: Task) => {
      if (!actor) throw new Error('Actor not available');
      await actor.updateTask(task);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });
}

export function useDeleteTask() {
  const { actor } = useActor();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (taskId: string) => {
      if (!actor) throw new Error('Actor not available');
      await actor.deleteTask(taskId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });
}

// Day Records Queries
export function useGetDayRecords() {
  const { actor, isFetching } = useActor();

  return useQuery<DayRecord[]>({
    queryKey: ['dayRecords'],
    queryFn: async () => {
      if (!actor) return [];
      return actor.getDayRecords();
    },
    enabled: !!actor && !isFetching,
  });
}

export function useAddDayRecord() {
  const { actor } = useActor();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (dayRecord: DayRecord) => {
      if (!actor) throw new Error('Actor not available');
      await actor.addDayRecord(dayRecord);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dayRecords'] });
    },
  });
}

// System Log Queries
export function useGetSystemLogs() {
  const { actor, isFetching } = useActor();

  return useQuery<SystemLogEntry[]>({
    queryKey: ['systemLogs'],
    queryFn: async () => {
      if (!actor) return [];
      return actor.getSystemLogs();
    },
    enabled: !!actor && !isFetching,
  });
}

export function useAddLogEntry() {
  const { actor } = useActor();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (logEntry: SystemLogEntry) => {
      if (!actor) throw new Error('Actor not available');
      await actor.addLogEntry(logEntry);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['systemLogs'] });
    },
  });
}

// XP State Queries
export function useGetXPState() {
  const { actor, isFetching } = useActor();

  return useQuery<XPState>({
    queryKey: ['xpState'],
    queryFn: async () => {
      if (!actor) throw new Error('Actor not available');
      return actor.getXPState();
    },
    enabled: !!actor && !isFetching,
  });
}

export function useUpdateXPState() {
  const { actor } = useActor();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (xpState: XPState) => {
      if (!actor) throw new Error('Actor not available');
      await actor.updateXPState(xpState);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['xpState'] });
    },
  });
}

// Rank Queries
export function useGetRank() {
  const { actor, isFetching } = useActor();

  return useQuery<Rank | null>({
    queryKey: ['rank'],
    queryFn: async () => {
      if (!actor) return null;
      return actor.getRank('current');
    },
    enabled: !!actor && !isFetching,
  });
}

export function useUpdateRank() {
  const { actor } = useActor();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (rank: Rank) => {
      if (!actor) throw new Error('Actor not available');
      await actor.updateRank(rank);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['rank'] });
    },
  });
}

// Lock-In Mode Queries
export function useGetLockInMode() {
  const { actor, isFetching } = useActor();

  return useQuery<LockInMode>({
    queryKey: ['lockInMode'],
    queryFn: async () => {
      if (!actor) throw new Error('Actor not available');
      return actor.getLockInMode();
    },
    enabled: !!actor && !isFetching,
  });
}

export function useUpdateLockInMode() {
  const { actor } = useActor();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (lockInMode: LockInMode) => {
      if (!actor) throw new Error('Actor not available');
      await actor.updateLockInMode(lockInMode);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['lockInMode'] });
    },
  });
}
