# Three completed card actions

**Goal:** After three successfully classified card actions finish, including the startup card, stop card presence checks, proximity stops and classification for the rest of that process. Keep lane following and the third action's existing hold/restoration unchanged.

**Design:** Track only event IDs dispatched from successful classifications. Count a fresh matching `action_done` once, including startup events whose command hold has already expired. Failed votes, pose events and unrelated or duplicate completion packets do not count. The counter resets with a new vision process.

**Execution:** Apply the writing-plans and executing-plans workflows in this session; the user has authorized implementation, GitHub push and SSH synchronization.

- [x] Add regression replays with continuously visible cards: exactly three ordinary actions, startup plus two ordinary actions, delayed completion, and failed votes.
- [x] Run the new tests and confirm the current code still attempts a fourth action (six ordinary events, or five events including startup, in regression replays).
- [x] Add a small completion tracker in `card_flow.py`; register startup and ordinary events in `run_policy_vision.py`, poll completion before detection and skip shape updates after three completions. Add count/enabled diagnostics and a transition log.
- [x] Run the card, startup and vision regression suites locally and on Jetson without starting robot motion. All selected Linux suites pass; Windows sandbox UDP and temporary-video-file checks require the Linux verification. Local lifecycle suite: 31 tests pass. Jetson card suite: 67 tests, one optional-fixture skip; vision suite: 117 tests pass; startup and button suites pass.
- [ ] Review the diff, commit only this change, push the current branch to GitHub, fast-forward the active Jetson repository and compare deployed files. Preserve Jetson's existing uncommitted two-second voting adjustment.

**Synchronization:** Local tracked/untracked edits were discarded at the user's request. Git history was fast-forwarded from local `af53f1a` to active Jetson `2611e4c`. The separate nested repository under `.diagnostics/stm32-health-review` is outside the cleanup of this repository. Do not restart the running robot processes during deployment.
