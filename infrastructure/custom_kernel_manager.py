from __future__ import annotations

from jupyter_server.services.kernels.kernelmanager import MappingKernelManager

SHARED_KERNEL_DIRS = ['notebooks/regression']


class CustomKernelManager(MappingKernelManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dir_to_kernel_id = {}

    async def start_kernel(self, kernel_id=None, path=None, **kwargs):
        self.log.info(f'start_kernel {kernel_id=} {path=} {kwargs=}')

        matched_dir = None
        for nb_dir in SHARED_KERNEL_DIRS:
            if str(path).startswith(nb_dir):
                matched_dir = nb_dir
                kernel_id = self.dir_to_kernel_id.get(nb_dir)
                self.log.info(
                    f'existing kernel for {matched_dir}: {kernel_id=}'
                )
                break
        kernel_id = await super().start_kernel(kernel_id, path, **kwargs)
        if matched_dir is not None:
            self.dir_to_kernel_id[matched_dir] = kernel_id
        self.log.info(f'result {kernel_id=}')
        return kernel_id


__all__ = ['CustomKernelManager']
