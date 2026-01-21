import Float "mo:core/Float";
import Map "mo:core/Map";
import Text "mo:core/Text";
import Nat "mo:core/Nat";
import Time "mo:core/Time";
import Int "mo:core/Int";
import List "mo:core/List";
import Order "mo:core/Order";
import Array "mo:core/Array";
import Iter "mo:core/Iter";
import Runtime "mo:core/Runtime";
import Principal "mo:core/Principal";
import Migration "migration";
import AccessControl "authorization/access-control";
import MixinAuthorization "authorization/MixinAuthorization";

// Migration setup
(with migration = Migration.run)
actor {
  let accessControlState = AccessControl.initState();
  include MixinAuthorization(accessControlState);

  module Task {
    public func compare(a : Task, b : Task) : Order.Order {
      switch (Text.compare(a.id, b.id)) {
        case (#equal) { Int.compare(a.dueDate, b.dueDate) };
        case (order) { order };
      };
    };
  };

  type TaskStatus = { #pending; #completed; #failed };

  type Task = {
    id : Text;
    title : Text;
    description : Text;
    category : ?Text;
    difficulty : Nat;
    plannedXP : Float;
    dueDate : Int;
    recurrence : ?Text;
    status : TaskStatus;
    completionFactor : Float;
    isCritical : Bool;
    owner : Principal;
  };

  type SeasonModifier = {
    decayCoefficient : Float;
    warmUpBuffer : Nat;
    xpYieldReduction : Float;
  };

  type Season = {
    id : Text;
    duration : Nat;
    startDate : Int;
    endDate : Int;
    modifiers : SeasonModifier;
    isActive : Bool;
  };

  type DayRecord = {
    date : Int;
    plannedXP : Float;
    completedXP : Float;
    earnedXP : Float;
    penaltiesApplied : Float;
    XPdebt : Float;
    owner : Principal;
  };

  type XPState = {
    totalXP : Float;
    dailyEarnedXP : Float;
    dailyLostXP : Float;
    debt : Float;
  };

  type Rank = {
    currentRank : Text;
    xpFloor : Float;
    xpCeiling : Float;
    surplusBuffer : Float;
    demotionFlag : Bool;
  };

  type SystemLogEntry = {
    timestamp : Int;
    day : Nat;
    eventType : Text;
    details : Text;
    xpChange : Float;
    rankChange : ?Text;
    owner : Principal;
  };

  type LockInMode = {
    isActive : Bool;
    startDate : Int;
    daysRemaining : Nat;
  };

  public type UserProfile = {
    name : Text;
    createdAt : Int;
  };

  type AdvancedAnalytics = {
    behavioralPatterns : [Text];
    ambitionCapacityIndex : Float;
    collapseRecoveryCurves : [Float];
    failureProbabilityFlags : [Nat];
    rankTrajectoryProjection : [Float];
  };

  type UserData = {
    xpState : XPState;
    lockInMode : LockInMode;
    dayCounter : Nat;
    currentRank : Rank;
    isInitialized : Bool;
    currentSeason : ?Season;
    analytics : AdvancedAnalytics;
  };

  // User-specific data stores
  let userProfiles = Map.empty<Principal, UserProfile>();
  let userDataStore = Map.empty<Principal, UserData>();
  let taskStore = Map.empty<Text, Task>();
  let dayStore = Map.empty<Text, DayRecord>();
  let systemLogStore = Map.empty<Text, SystemLogEntry>();

  public type CreateTaskData = {
    id : Text;
    title : Text;
    description : Text;
    category : ?Text;
    difficulty : Nat;
    plannedXP : Float;
    dueDate : Int;
    recurrence : ?Text;
    isCritical : Bool;
  };

  public type TaskLightningResponse = Task;
  public type DayLightningResponse = {
    plannedXP : Float;
    completedXP : Float;
    earnedXP : Float;
    penaltiesApplied : Float;
    XPdebt : Float;
  };

  public type InitTaskLightning = {
    plannedXP : Float;
    completedXP : Float;
    earnedXP : Float;
    penaltiesApplied : Float;
    XPdebt : Float;
  };

  // User Profile Management
  public query ({ caller }) func getCallerUserProfile() : async ?UserProfile {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can view profiles");
    };
    userProfiles.get(caller);
  };

  public query ({ caller }) func getUserProfile(user : Principal) : async ?UserProfile {
    if (caller != user and not AccessControl.isAdmin(accessControlState, caller)) {
      Runtime.trap("Unauthorized: Can only view your own profile or admin access required");
    };
    userProfiles.get(user);
  };

  public shared ({ caller }) func saveCallerUserProfile(profile : UserProfile) : async () {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can save profiles");
    };
    userProfiles.add(caller, profile);
  };

  // Helper function to get or create user data
  func getUserData(user : Principal) : UserData {
    switch (userDataStore.get(user)) {
      case (?data) { data };
      case (null) {
        let initialData : UserData = {
          xpState = {
            totalXP = 0;
            dailyEarnedXP = 0;
            dailyLostXP = 0;
            debt = 0;
          };
          lockInMode = {
            isActive = false;
            startDate = Time.now();
            daysRemaining = 30;
          };
          dayCounter = 0;
          currentRank = {
            currentRank = "Novice";
            xpFloor = 0;
            xpCeiling = 1000;
            surplusBuffer = 300;
            demotionFlag = false;
          };
          isInitialized = false;
          currentSeason = null;
          analytics = {
            behavioralPatterns = [];
            ambitionCapacityIndex = 0.0;
            collapseRecoveryCurves = [];
            failureProbabilityFlags = [];
            rankTrajectoryProjection = [];
          };
        };
        userDataStore.add(user, initialData);
        initialData;
      };
    };
  };

  func updateUserData(user : Principal, data : UserData) {
    userDataStore.add(user, data);
  };

  public shared ({ caller }) func initializeTask(_ : CreateTaskData) : async () {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can initialize tasks");
    };

    var userData = getUserData(caller);
    if (userData.isInitialized) {
      return;
    };

    let dayKey = caller.toText() # "_" # userData.dayCounter.toText();
    let initialDayRecord : DayRecord = {
      date = Time.now();
      plannedXP = 0.0;
      completedXP = 0.0;
      earnedXP = 0.0;
      penaltiesApplied = 0.0;
      XPdebt = 0.0;
      owner = caller;
    };

    dayStore.add(dayKey, initialDayRecord);

    let taskKey = caller.toText() # "_initial_task";
    let initialTask : Task = {
      id = "initial_task";
      title = "Welcome Task";
      description = "This is your first task in LifeQuest";
      category = ?"Onboarding";
      difficulty = 1;
      plannedXP = 0.0;
      dueDate = Time.now();
      recurrence = null;
      status = #pending;
      completionFactor = 0.0;
      isCritical = false;
      owner = caller;
    };
    taskStore.add(taskKey, initialTask);

    userData := {
      xpState = userData.xpState;
      lockInMode = userData.lockInMode;
      dayCounter = userData.dayCounter + 1;
      currentRank = userData.currentRank;
      isInitialized = true;
      currentSeason = userData.currentSeason;
      analytics = userData.analytics;
    };
    updateUserData(caller, userData);
  };

  public shared ({ caller }) func createTask(taskData : CreateTaskData) : async () {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can create tasks");
    };
    createTaskSync(caller, taskData);
  };

  func createTaskSync(owner : Principal, taskData : CreateTaskData) {
    let taskKey = owner.toText() # "_" # taskData.id;
    let newTask : Task = {
      id = taskData.id;
      title = taskData.title;
      description = taskData.description;
      category = taskData.category;
      difficulty = taskData.difficulty;
      plannedXP = taskData.plannedXP;
      dueDate = taskData.dueDate;
      recurrence = taskData.recurrence;
      status = #pending;
      completionFactor = 0.0;
      isCritical = taskData.isCritical;
      owner = owner;
    };

    taskStore.add(taskKey, newTask);
  };

  func deleteTaskSync(owner : Principal, taskId : Text) {
    let taskKey = owner.toText() # "_" # taskId;
    taskStore.remove(taskKey);
  };

  public shared ({ caller }) func deleteTask(taskId : Text) : async () {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can delete tasks");
    };
    deleteTaskSync(caller, taskId);
  };

  func updateTaskSync(owner : Principal, taskData : Task) {
    let taskKey = owner.toText() # "_" # taskData.id;
    switch (taskStore.get(taskKey)) {
      case (null) { Runtime.trap("Task not found") };
      case (?existingTask) {
        if (existingTask.owner != owner) {
          Runtime.trap("Unauthorized: Cannot update another user's task");
        };
        taskStore.add(taskKey, taskData);
      };
    };
  };

  public shared ({ caller }) func updateTask(taskData : Task) : async () {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can update tasks");
    };
    updateTaskSync(caller, taskData);
  };

  public shared ({ caller }) func addDayRecord(dayRecord : DayRecord) : async () {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can add day records");
    };

    var userData = getUserData(caller);
    let dayKey = caller.toText() # "_" # userData.dayCounter.toText();
    let recordWithOwner : DayRecord = {
      date = dayRecord.date;
      plannedXP = dayRecord.plannedXP;
      completedXP = dayRecord.completedXP;
      earnedXP = dayRecord.earnedXP;
      penaltiesApplied = dayRecord.penaltiesApplied;
      XPdebt = dayRecord.XPdebt;
      owner = caller;
    };
    dayStore.add(dayKey, recordWithOwner);
    userData := {
      xpState = userData.xpState;
      lockInMode = userData.lockInMode;
      dayCounter = userData.dayCounter + 1;
      currentRank = userData.currentRank;
      isInitialized = userData.isInitialized;
      currentSeason = userData.currentSeason;
      analytics = userData.analytics;
    };
    updateUserData(caller, userData);
  };

  public shared ({ caller }) func addLogEntry(logEntry : SystemLogEntry) : async () {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can add log entries");
    };

    let logKey = caller.toText() # "_" # logEntry.eventType # "_" # logEntry.timestamp.toText();
    let entryWithOwner : SystemLogEntry = {
      timestamp = logEntry.timestamp;
      day = logEntry.day;
      eventType = logEntry.eventType;
      details = logEntry.details;
      xpChange = logEntry.xpChange;
      rankChange = logEntry.rankChange;
      owner = caller;
    };

    if (logEntry.eventType == "DELETED") {
      systemLogStore.remove(logKey);
    } else {
      systemLogStore.add(logKey, entryWithOwner);
    };
  };

  public query ({ caller }) func getTasks() : async [Task] {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can view tasks");
    };

    let allTasks = taskStore.values().toArray();
    let userTasks = allTasks.filter(
      func(task : Task) : Bool {
        task.owner == caller;
      }
    );
    userTasks;
  };

  public query ({ caller }) func getDayRecords() : async [DayRecord] {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can view day records");
    };

    let allRecords = dayStore.values().toArray();
    let userRecords = allRecords.filter(
      func(record : DayRecord) : Bool {
        record.owner == caller;
      }
    );
    userRecords;
  };

  public query ({ caller }) func getSystemLogs() : async [SystemLogEntry] {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can view system logs");
    };

    let allLogs = systemLogStore.values().toArray();
    let userLogs = allLogs.filter(
      func(log : SystemLogEntry) : Bool {
        log.owner == caller;
      }
    );
    userLogs;
  };

  public query ({ caller }) func getRank(rankType : Text) : async ?Rank {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can view rank");
    };

    if (rankType == "null") { Runtime.trap("No rank provided") };
    let userData = getUserData(caller);
    ?userData.currentRank;
  };

  public shared ({ caller }) func getLightningTask() : async [TaskLightningResponse] {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can view lightning tasks");
    };

    let allTasks = taskStore.values().toArray();
    let userTasks = allTasks.filter(
      func(task : Task) : Bool {
        task.owner == caller;
      }
    );
    let sortedTasks = userTasks.sort();
    let resultArray = sortedTasks.sliceToArray(0, Nat.min(5, sortedTasks.size()));
    if (resultArray.size() == 0) { Runtime.trap("No task found") };
    resultArray;
  };

  public query ({ caller }) func getXPState() : async XPState {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can view XP state");
    };

    let userData = getUserData(caller);
    userData.xpState;
  };

  public shared ({ caller }) func updateXPState(newXPState : XPState) : async () {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can update XP state");
    };

    var userData = getUserData(caller);
    userData := {
      xpState = newXPState;
      lockInMode = userData.lockInMode;
      dayCounter = userData.dayCounter;
      currentRank = userData.currentRank;
      isInitialized = userData.isInitialized;
      currentSeason = userData.currentSeason;
      analytics = userData.analytics;
    };
    updateUserData(caller, userData);
  };

  public query ({ caller }) func getLockInMode() : async LockInMode {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can view lock-in mode");
    };

    let userData = getUserData(caller);
    userData.lockInMode;
  };

  public shared ({ caller }) func updateLockInMode(newLockInMode : LockInMode) : async () {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can update lock-in mode");
    };

    var userData = getUserData(caller);
    userData := {
      xpState = userData.xpState;
      lockInMode = newLockInMode;
      dayCounter = userData.dayCounter;
      currentRank = userData.currentRank;
      isInitialized = userData.isInitialized;
      currentSeason = userData.currentSeason;
      analytics = userData.analytics;
    };
    updateUserData(caller, userData);
  };

  public shared ({ caller }) func updateRank(newRank : Rank) : async () {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can update rank");
    };

    var userData = getUserData(caller);
    userData := {
      xpState = userData.xpState;
      lockInMode = userData.lockInMode;
      dayCounter = userData.dayCounter;
      currentRank = newRank;
      isInitialized = userData.isInitialized;
      currentSeason = userData.currentSeason;
      analytics = userData.analytics;
    };
    updateUserData(caller, userData);
  };
};
