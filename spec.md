# LifeQuest   Analytical Task Management System

## Overview
LifeQuest is a brutally honest, analytical task management application that tracks daily XP performance through a punitive accountability system. The application emphasizes consequences over motivation, using a cold dashboard aesthetic with no gamification elements.

## Design System
- **Color Palette**: Background `#0B0E11`, primary text `#E6E6E6`, secondary `#8A8F98`, accent `#B6FF3B`, negative dull red
- **Typography**: Sans-serif (Inter/IBM Plex) with strict hierarchy
- **Visual Style**: Cold system dashboard aesthetic with no rounded corners, animations, or decorative elements

## Core Features

### Task Management (Quests)
- Create tasks with title, description, category, difficulty, planned XP, due date, and optional recurrence
- Track completion status with completion factor (0-1 scale)
- Mark tasks as "critical" with commitment locks preventing deletion or XP/difficulty reduction
- Tasks are stored in the backend with full CRUD operations

### XP Calculation System
Daily XP calculation follows the formula:
`XP_d = XP_{d-1} + Earned_XP_d - Miss_Penalty_d - Avoidance_Decay_d - Debt_Leak_d`

- **Earned XP**: Sum of (Task XP × Completion Factor)
- **Miss Penalties**: Escalating penalties (1 day = 10%, up to 5+ days = 60%)
- **Avoidance Decay**: Applied when daily completion ≤40% or tasks rescheduled, calculated as (Planned_XP - Completed_XP) × decay factor (0.4-0.6)
- **XP Debt**: Accumulates from unfinished tasks, leaks daily by 15%

### Rank System
- Superlinear XP requirements (base × r^1.35)
- Each rank has surplus buffer (300-500 XP)
- Immediate demotion when buffer exhausted
- Fragile rank status with no grace periods

### Warm-Up Phase
- First 7 days: XP loss capped at 10-15%
- Rank demotion disabled during warm-up
- Clear labeling of warm-up buffer status

### Performance Visualization
- 100-day performance graph with normalized performance score
- EMA or Bezier-smoothed line visualization
- Subtle horizontal rank bands for context
- Downward slopes emphasized over upward trends
- Custom SVG/Canvas implementation (no charting libraries)

### System Integrity Log
- Chronological plain text log of all XP events
- Exact mathematical calculations displayed
- All penalties, demotions, and rank changes recorded

### Lock-In Mode
- 30-day irreversible commitment mode
- Freezes all XP and rank calculation rules
- Cannot be disabled once activated

### Manual Failure Declaration
- "Conscious Failure" marking option
- Applies standard penalties with reduced avoidance decay
- Explicit logging of declared failures

## Advanced Analytics Layer

### Behavioral Pattern Detection
- Analyze completed and failed tasks by category, time of day, weekday, and difficulty
- Calculate failure rates and detect patterns
- Display factual statements like "Critical tasks fail 71% after 8:30 PM"
- Backend aggregates and calculates these statistics
- Frontend "Analytics" tab shows statements in stark text-only presentation

### Ambition vs Capacity Index
- Calculate ratio of planned vs completed XP and volatility (standard deviation of completion)
- Classify users as under-ambitious, calibrated, or chronically overcommitting
- Display quietly under user profile and analytics panel

### Collapse & Recovery Curves
- Track XP decay and recovery speed after missed days
- Calculate discipline half-life values
- Visualize as curve on analytics graph

### Failure Probability Flags
- Use performance history to predict task failure likelihood (0-100%)
- Show as number only beside each task, no suggestions

### Rank Trajectory Projection
- Predictive model projecting rank trend based on recent XP trends
- Present faint projection line on the 100-day graph

### What-If Simulator
- Interactive testing of changes in planned XP, task removal, or schedule shifts
- Graph updates instantly to reflect projected XP trajectory changes

## Multi-Season System

### Season Structure
- Backend structures for Season with 60/90/120-day durations
- Immutable logs and historical archiving of completed seasons
- Seasonal modifiers system allowing self-imposed difficulty increases
- Once chosen, seasonal modifiers are locked for the entire season

### Career Profile
- Summary stats showing seasons played, average rank, worst collapse, longest recovery, total XP lost
- Historical performance across all completed seasons

## Discipline Enforcement

### Time-Locked Planning
- Daily planning cutoff time configuration
- Planning after cutoff increases avoidance penalty coefficient

### Commitment Locks
- Critical task marking prevents deletion or XP/difficulty reduction
- Harsher penalties for incomplete critical tasks

### Capacity Auto-Correction
- Track overplanning patterns
- Dynamically restrict allowed planning XP until recovery for repeated failures

## Data Integrity & Trust

### Tamper Detection
- Log and flag backdated tasks, mass edits, or irregular timestamp entries
- Informational only, no penalties applied

### Immutable Audit Mode
- Permanently lock logs and history once enabled
- Cannot be reversed after activation

### Full Data Export
- Raw download of XP logs, daily records, rank transitions, and performance graph values
- Available in CSV and JSON formats

## Visualization & UX Enhancements

### Graph Overlays
- Compare seasons, best/worst 30-day periods, and planned vs actual XP performance
- Multiple overlay options on the main performance graph

### Failure Cluster Heatmaps
- Visualize failure rates by category, day type, and time blocks
- Cold gray-to-red grid heatmaps showing failure patterns

## Data Storage

### Backend Data
- **Tasks**: Complete task records with all metadata including critical flags
- **Daily Records**: Daily XP calculations, penalties, and performance metrics
- **User Progress**: Total XP, current rank, debt status
- **System Log**: All XP events and calculations with tamper detection
- **Lock-In Status**: Current mode and remaining days
- **Season Data**: Historical season records and current season progress
- **Analytics Data**: Behavioral patterns, failure rates, and statistical calculations
- **Audit Trail**: Immutable history when audit mode is enabled

### Frontend State
- Current day calculations
- Graph rendering data with overlays
- UI state management
- Analytics visualization state
- What-if simulator state
- Offline data synchronization

## Technical Requirements
- React functional components with Redux or Zustand state management
- Custom SVG/Canvas charts only
- CSS variables and Tailwind for strict color adherence
- IndexedDB for offline-first architecture
- PWA-ready implementation
- Performance prioritized over visual decoration

## Success Metrics
- User retention beyond 30 days
- Accurate demotion tracking
- System described as "uncomfortable but accurate"
- No social features, achievements, or motivational elements
