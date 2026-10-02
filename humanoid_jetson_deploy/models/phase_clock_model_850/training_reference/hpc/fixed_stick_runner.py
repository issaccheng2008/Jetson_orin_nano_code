"""Save clean-crossing readiness beside each policy checkpoint."""
from rsl_rl.runners import OnPolicyRunner


class FixedStickRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        saved_infos = dict(infos or {})
        state = self.env.unwrapped._fixed_stick_state
        saved_infos['fixed_stick_training'] = state.clean_stats.summary()
        saved_infos['fixed_stick_control'] = state.control_settings()
        return super().save(path, infos=saved_infos)
