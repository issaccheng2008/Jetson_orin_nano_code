# Remote / 454c89b merge verification

Merge parents: `2725ac1ac40f0028d7aa6fde53edaa632ab6f2b5` and `454c89bbe3c77d26d620f11ec978b776a91f60e3`.

- Resolved README by retaining both the remote introduction and the latest cue-only vision description.
- Preserved the latest runtime safety, diagnostics, scripted/curve modes, and vision card lifecycle. Added remote as an independent UDP source on port 5007; vision remains on 5005.
- Card controllers, posture metadata, startup gates and attitude broadcasting remain vision-only. Remote does not publish attitude into its own receive port.
- Added regression coverage for immediate remote zero despite minimum walking hold, incompatible fixed frames, and no remote card-controller/attitude startup.
- Runtime suite: 183 tests passed. Repository suite: 829 run, OK with 22 environment-dependent skips. Frontend suite: 11 passed. Browser: 21 checks passed.
- The initial environment lacked matplotlib and `tee`; test dependencies and Git's existing `tee` were supplied only to the test environment. A pre-existing real-clock phase test initially exceeded its inference-count assertion; its isolated rerun and the final full suite passed without changing its implementation or assertions.
- Independent review found no actionable merge regression. Hardware motion is unverified.
- Merge was completed in a managed worktree to preserve the original directory's unrelated uncommitted work.
