import time
import psutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.data = {
            'timestamp': [],
            'cpu_usage_percent': [],
            'io_write_mb': [],          
            'io_write_rate_mbps': []    
        }
        self.last_io_write = None
        self.last_timestamp = None

    def _get_io_writes_mb(self):
        try:
            io = self.process.io_counters()
            return io.write_bytes / (1024 * 1024)
        except (psutil.AccessDenied, AttributeError):
            return 0.0

    def start(self):
        self.last_timestamp = time.time()
        self.last_io_write = self._get_io_writes_mb()
        self.record()   

    def record(self):
        now = time.time()
        cpu_percent = self.process.cpu_percent(interval=None)  # интервал = 0 – моментальный замер
        current_io_mb = self._get_io_writes_mb()
        rate_mbps = 0.0
        if self.last_io_write is not None and self.last_timestamp is not None:
            delta_time = now - self.last_timestamp
            if delta_time > 0:
                delta_mb = current_io_mb - self.last_io_write
                rate_mbps = delta_mb / delta_time

        self.data['timestamp'].append(now)
        self.data['cpu_usage_percent'].append(cpu_percent)
        self.data['io_write_mb'].append(current_io_mb)
        self.data['io_write_rate_mbps'].append(rate_mbps)

        self.last_timestamp = now
        self.last_io_write = current_io_mb

    def end(self, save_path='records/resource_plot.png', show=False):
        df = pd.DataFrame(self.data)
        start_time = df['timestamp'].iloc[0]
        df['time_sec'] = df['timestamp'] - start_time

        sns.set_theme(style="darkgrid", context="talk")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Использование ресурсов процесса', fontsize=14)

        ax = axes[0]
        sns.lineplot(x='time_sec', y='cpu_usage_percent', data=df, ax=ax, color='blue')
        ax.set_title('Загрузка CPU (%)')
        ax.set_ylabel('%')
        ax.set_xlabel('Время (с)')

        ax = axes[1]
        sns.lineplot(x='time_sec', y='io_write_rate_mbps', data=df, ax=ax, color='green')
        ax.set_title('Скорость записи на диск (МБ/с)')
        ax.set_ylabel('МБ/с')
        ax.set_xlabel('Время (с)')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"График сохранён как {save_path}")
        if show:
            plt.show()

        return df

if __name__ == "__main__":
    mon = ResourceMonitor()
    mon.start()

    import numpy as np
    for i in range(30):
        _ = np.random.randn(1000, 1000).dot(np.random.randn(1000, 1000))
        with open(f'temp_{i}.txt', 'w') as f:
            f.write('x' * 10_000_000)   
        mon.record()
        time.sleep(0.2)

    mon.end()