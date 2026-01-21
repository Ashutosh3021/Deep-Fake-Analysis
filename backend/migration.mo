import Map "mo:core/Map";
import Text "mo:core/Text";
import Nat "mo:core/Nat";
import Time "mo:core/Time";
import Principal "mo:core/Principal";

module {
  // Old Types
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

  type Task = {
    id : Text;
    title : Text;
    description : Text;
    category : ?Text;
    difficulty : Nat;
    plannedXP : Float;
    dueDate : Int;
    recurrence : ?Text;
    status : { #pending; #completed; #failed };
    completionFactor : Float;
    owner : Principal;
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

  type OldUserData = {
    xpState : XPState;
    lockInMode : LockInMode;
    dayCounter : Nat;
    currentRank : Rank;
    isInitialized : Bool;
  };

  type OldActor = {
    userProfiles : Map.Map<Principal, UserProfile>;
    userDataStore : Map.Map<Principal, OldUserData>;
    taskStore : Map.Map<Text, Task>;
    dayStore : Map.Map<Text, DayRecord>;
    systemLogStore : Map.Map<Text, SystemLogEntry>;
  };

  type AdvancedAnalytics = {
    behavioralPatterns : [Text];
    ambitionCapacityIndex : Float;
    collapseRecoveryCurves : [Float];
    failureProbabilityFlags : [Nat];
    rankTrajectoryProjection : [Float];
  };

  type NewTask = {
    id : Text;
    title : Text;
    description : Text;
    category : ?Text;
    difficulty : Nat;
    plannedXP : Float;
    dueDate : Int;
    recurrence : ?Text;
    status : { #pending; #completed; #failed };
    completionFactor : Float;
    isCritical : Bool;
    owner : Principal;
  };

  type NewUserData = {
    xpState : XPState;
    lockInMode : LockInMode;
    dayCounter : Nat;
    currentRank : Rank;
    isInitialized : Bool;
    currentSeason : ?Season;
    analytics : AdvancedAnalytics;
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

  type NewActor = {
    userProfiles : Map.Map<Principal, UserProfile>;
    userDataStore : Map.Map<Principal, NewUserData>;
    taskStore : Map.Map<Text, NewTask>;
    dayStore : Map.Map<Text, DayRecord>;
    systemLogStore : Map.Map<Text, SystemLogEntry>;
  };

  func createAnalytics() : AdvancedAnalytics {
    {
      behavioralPatterns = [];
      ambitionCapacityIndex = 0.0;
      collapseRecoveryCurves = [];
      failureProbabilityFlags = [];
      rankTrajectoryProjection = [];
    };
  };

  func convertUserData(oldUserData : OldUserData) : NewUserData {
    {
      oldUserData with
      currentSeason = null;
      analytics = createAnalytics();
    };
  };

  func convertTask(oldTask : Task) : NewTask {
    {
      oldTask with
      isCritical = false;
    };
  };

  public func run(old : OldActor) : NewActor {
    let newUserDataStore = old.userDataStore.map<Principal, OldUserData, NewUserData>(
      func(_principal, oldUserData) {
        convertUserData(oldUserData);
      }
    );
    let newTaskStore = old.taskStore.map<Text, Task, NewTask>(
      func(_id, oldTask) {
        convertTask(oldTask);
      }
    );
    { old with userDataStore = newUserDataStore; taskStore = newTaskStore };
  };
};
