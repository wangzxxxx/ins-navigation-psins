import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import time

def plot_progress(logdir="D:/psins251010/psins251010/psins_py/rl_tensorboard"):
    # 找到最新的 PPO_x 文件夹
    subdirs = [os.path.join(logdir, d) for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))]
    latest_dir = max(subdirs, key=os.path.getmtime)
    
    # 加载数据
    ea = EventAccumulator(latest_dir)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    
    if 'custom/episode_worst_red' not in tags:
        print(f"数据尚未生成或还在加载中：{latest_dir}")
        return False
        
    worst_red_events = ea.Scalars('custom/episode_worst_red')
    time_events = ea.Scalars('custom/episode_time')
    
    # 提取 x (step/episode) 和 y
    red_y = [e.value for e in worst_red_events]
    time_y = [e.value for e in time_events]
    episodes = list(range(1, len(red_y) + 1))
    
    plt.figure(figsize=(10, 5))
    
    # 左侧 Y 轴：最差收敛指标 (%)
    ax1 = plt.gca()
    ax1.plot(episodes, red_y, 'b-', marker='o', label='Worst Reduction % (Lower is Better)')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Worst Reduction (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    
    # 右侧 Y 轴：耗时 (s)
    ax2 = ax1.twinx()
    ax2.plot(episodes, time_y, 'r--', marker='s', label='Episode Time (s)')
    ax2.set_ylabel('Total Trajectory Time (s)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title(f'PPO Training Progress (latest: {os.path.basename(latest_dir)})')
    plt.tight_layout()
    plt.show()
    return True

if __name__ == "__main__":
    print("Reading Tensorboard logs from Python directly...")
    success = plot_progress()
    if not success:
        print("Waiting for the first episode to finish and write logs...")
