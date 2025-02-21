from ti.policy.base_image_policy import BaseImagePolicy
from ti.env_runner.base_image_runner import BaseImageRunner

class RealRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            **kwargs,):
        super().__init__(output_dir)
    
    def run(self, policy: BaseImagePolicy):
        return dict()
