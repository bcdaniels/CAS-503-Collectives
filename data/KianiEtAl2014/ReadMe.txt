
Each dataset comes from one session of recordings and includes the following variables:

1. trial_info.coherence            Motion strength (coherence) on each trial
2. trial_info.chosen_target        The chosen target on each trial
3. trial_info.correct_target       Was the trial rewarded or not? Note that for 0 motion coherence the rewarded target was chosen randomly.
4. event_times.dots_onset          Dots onset time on each trial. All events and spike times are aligned to the onset of random dots 
5. event_times.dots_offset         Dots offset time (beginning of delay period) on each trial 
6. event_times.go_cue              Time of Go cue (offset of fixation point) on each trial 
7. event_times.sacc_onset          Saccade onset on each trial 
8. spike_times                     A 2D matrix that contains spike times and of all recorded units. Each row corresponds to a trial and each column to a unit. Spike times are with respect to the motion onset.

