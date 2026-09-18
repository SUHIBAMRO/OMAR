"""Live GPU-memory-over-time monitoring, for notebooks that want a real
utilization timeline (Omar's own request, 2026-09-18) rather than just a
single "peak memory" number -- none of the round-12 retrains tracked
this (confirmed by reading train_B1.py's own single-resolution training
loop, which DOES track a per-epoch `gpu_peak_mem_device_mb`, against
resolution_invariance_zeroshot.py's own multi-resolution training loop,
which does not -- every real retrain and the direct-N1401 ablation both
went through the latter, so no run this project has ever produced has
this data; it cannot be recovered without re-running training, which
Timon himself said not to bother with here).

Samples torch.cuda.mem_get_info() (total - free), the same "how much of
the whole GPU is in use" quantity nvidia-smi itself reports and
train_B1.py's own gpu_peak_mem_device_mb already uses -- not
max_memory_allocated/reserved, which are PyTorch's own allocator
bookkeeping and exclude the CUDA context's fixed overhead.
"""
import threading
import time

import torch


class GPUMemoryMonitor:
    """Background thread that samples GPU memory every `interval_s`
    seconds while active. Use as a context manager around any block of
    real GPU work:

        with GPUMemoryMonitor(device) as mon:
            ... real GPU work ...
        mon.save_plot('/path/to/fig.png', title='...')

    `mon.samples` is a list of (elapsed_seconds, used_mb) tuples;
    `mon.device_total_mb` is the device's own fixed total.
    `mon.mark(label)` can be called from the main thread between stages
    (e.g. between cases in a loop) to record a labeled vertical line on
    the eventual plot, without needing a second thread of its own.
    """

    def __init__(self, device, interval_s=1.0):
        self.device = device
        self.interval_s = interval_s
        self.samples = []
        self.marks = []  # (elapsed_seconds, label)
        self.device_total_mb = (
            torch.cuda.mem_get_info(device)[1] / 1e6 if device.type == "cuda" else 0.0)
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None

    def _run(self):
        while not self._stop.is_set():
            if self.device.type == "cuda":
                free_b, total_b = torch.cuda.mem_get_info(self.device)
                used_mb = (total_b - free_b) / 1e6
            else:
                used_mb = 0.0
            self.samples.append((time.time() - self._t0, used_mb))
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)
        return False

    def mark(self, label):
        """Call from the main thread (NOT from inside the sampling
        thread) to record a labeled point in time, e.g. the boundary
        between two cases in a loop."""
        self.marks.append((time.time() - self._t0, label))

    def peak_mb(self):
        return max((mb for _, mb in self.samples), default=0.0)

    def save_plot(self, out_path, title='GPU memory usage over time'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        try:
            from plot_style import PRIMARY
        except ImportError:
            PRIMARY = '#2E86AB'

        if not self.samples:
            print(f'[gpu_memory_monitor] no samples collected -- skipping {out_path}')
            return None

        ts = [s[0] for s in self.samples]
        mbs = [s[1] for s in self.samples]
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=200)
        ax.plot(ts, mbs, color=PRIMARY, linewidth=1.2)
        ax.fill_between(ts, mbs, color=PRIMARY, alpha=0.15)
        ax.axhline(self.device_total_mb, color='#7f7f7f', linestyle=':', linewidth=1,
                   label=f'device total ({self.device_total_mb / 1024:.1f} GB)')
        peak = self.peak_mb()
        ax.axhline(peak, color='#C0392B', linestyle='--', linewidth=1,
                   label=f'peak observed ({peak / 1024:.2f} GB)')
        for t_mark, label in self.marks:
            ax.axvline(t_mark, color='#999999', linewidth=0.6, alpha=0.6)
            ax.text(t_mark, ax.get_ylim()[1], label, rotation=90, fontsize=6,
                   va='top', ha='right', color='#555555')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('GPU memory in use (MB)')
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight')
        print(f'[gpu_memory_monitor] saved {out_path} ({len(self.samples)} samples, '
              f'peak={peak:.1f}MB / {self.device_total_mb:.1f}MB)')
        return out_path
