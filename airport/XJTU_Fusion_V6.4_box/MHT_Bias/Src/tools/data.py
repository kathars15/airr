# compare_tracks.py
"""
对比原始雷达航迹和MHT处理后的航迹
通过时间戳和空间距离进行匹配
"""

import json
import csv
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def load_raw_tracks(csv_file='raw_tracks.csv'):
    """加载原始雷达航迹"""
    try:
        df = pd.read_csv(csv_file)
        # 转换时间戳
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'])
        print(f"加载原始数据: {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"加载原始数据失败: {e}")
        return None

def load_mht_tracks(json_file='track_results.json'):
    """加载MHT处理后的航迹"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 处理JSON数组格式
            if content.startswith('['):
                data = json.loads(content)
            else:
                # 如果是每行一个JSON的格式
                lines = content.strip().split('\n')
                data = []
                for line in lines:
                    line = line.strip()
                    if line.endswith(','):
                        line = line[:-1]
                    if line and line not in ['[', ']', '[', ']']:
                        try:
                            data.append(json.loads(line))
                        except:
                            pass
        
        # 提取所有目标
        tracks = []
        for frame in data:
            if isinstance(frame, dict) and 'targets' in frame:
                timestamp = frame.get('timestamp', 0)
                for target in frame['targets']:
                    target['frame_timestamp'] = timestamp
                    tracks.append(target)
        
        print(f"加载MHT数据: {len(tracks)} 条记录")
        return tracks
    except Exception as e:
        print(f"加载MHT数据失败: {e}")
        return []

def match_tracks(raw_df, mht_tracks, time_window=2.0, distance_threshold=100):
    """
    匹配原始航迹和MHT航迹
    匹配条件：
    1. 时间差在 time_window 秒内
    2. 空间距离在 distance_threshold 米内
    """
    matches = []
    
    for _, raw in raw_df.iterrows():
        raw_time = raw['timestamp']
        raw_range = raw.get('range', 0)
        raw_azimuth = raw.get('azimuth', 0)
        
        # 寻找最匹配的MHT目标
        best_match = None
        best_distance = float('inf')
        
        for mht in mht_tracks:
            mht_time = mht.get('frame_timestamp', 0)
            mht_range = mht.get('range', 0)
            mht_azimuth = mht.get('azimuth', 0)
            
            # 时间差检查
            time_diff = abs(raw_time - mht_time)
            if time_diff > time_window:
                continue
            
            # 距离差
            range_diff = abs(raw_range - mht_range)
            if range_diff > distance_threshold:
                continue
            
            # 方位角差（考虑360度环绕）
            az_diff = abs(raw_azimuth - mht_azimuth)
            az_diff = min(az_diff, 360 - az_diff)
            if az_diff > 30:  # 方位角差不超过30度
                continue
            
            # 综合距离（加权）
            combined_dist = range_diff + az_diff * 10  # 方位角差权重10米/度
            
            if combined_dist < best_distance:
                best_distance = combined_dist
                best_match = mht
        
        if best_match:
            matches.append({
                'raw_time': raw_time,
                'raw_track_id': raw['track_id'],
                'raw_range': raw_range,
                'raw_azimuth': raw_azimuth,
                'mht_track_id': best_match.get('track_id'),
                'mht_range': best_match.get('range', 0),
                'mht_azimuth': best_match.get('azimuth', 0),
                'time_diff': abs(raw_time - best_match.get('frame_timestamp', 0)),
                'range_diff': abs(raw_range - best_match.get('range', 0)),
                'az_diff': min(abs(raw_azimuth - best_match.get('azimuth', 0)), 
                              360 - abs(raw_azimuth - best_match.get('azimuth', 0)))
            })
    
    return matches

def print_comparison_stats(matches, raw_df, mht_tracks):
    """打印对比统计信息"""
    print("\n" + "="*60)
    print("对比统计")
    print("="*60)
    
    print(f"\n原始航迹总数: {len(raw_df)}")
    print(f"MHT航迹总数: {len(mht_tracks)}")
    print(f"成功匹配数: {len(matches)}")
    
    if len(matches) > 0:
        # 计算平均差异
        avg_range_diff = np.mean([m['range_diff'] for m in matches])
        avg_az_diff = np.mean([m['az_diff'] for m in matches])
        avg_time_diff = np.mean([m['time_diff'] for m in matches])
        
        print(f"\n匹配结果统计:")
        print(f"  平均距离差: {avg_range_diff:.1f} 米")
        print(f"  平均方位角差: {avg_az_diff:.1f} 度")
        print(f"  平均时间差: {avg_time_diff:.2f} 秒")
        
        # 按MHT航迹ID分组
        mht_ids = set([m['mht_track_id'] for m in matches])
        print(f"\n匹配到的MHT航迹ID: {sorted(mht_ids)}")
        
        # 显示前10个匹配
        print("\n前10个匹配结果:")
        print("-"*80)
        print(f"{'原始ID':<10} {'MHT ID':<15} {'距离差(m)':<12} {'方位差(°)':<12} {'时间差(s)':<12}")
        print("-"*80)
        for m in matches[:10]:
            print(f"{m['raw_track_id']:<10} {m['mht_track_id']:<15} "
                  f"{m['range_diff']:<12.1f} {m['az_diff']:<12.1f} {m['time_diff']:<12.2f}")

def plot_comparison(matches, raw_df, mht_tracks):
    """绘制对比图"""
    if len(matches) == 0:
        print("没有匹配数据，无法绘图")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 提取匹配的数据
    raw_ranges = [m['raw_range'] for m in matches]
    mht_ranges = [m['mht_range'] for m in matches]
    raw_azimuths = [m['raw_azimuth'] for m in matches]
    mht_azimuths = [m['mht_azimuth'] for m in matches]
    
    # 1. 距离对比散点图
    ax = axes[0, 0]
    ax.scatter(raw_ranges, mht_ranges, alpha=0.5)
    ax.plot([0, max(raw_ranges)], [0, max(raw_ranges)], 'r--', label='理想线')
    ax.set_xlabel('原始距离 (m)')
    ax.set_ylabel('MHT距离 (m)')
    ax.set_title('距离对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 方位角对比散点图
    ax = axes[0, 1]
    ax.scatter(raw_azimuths, mht_azimuths, alpha=0.5)
    ax.plot([0, 360], [0, 360], 'r--', label='理想线')
    ax.set_xlabel('原始方位角 (度)')
    ax.set_ylabel('MHT方位角 (度)')
    ax.set_title('方位角对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 距离差分布直方图
    ax = axes[1, 0]
    range_diffs = [m['range_diff'] for m in matches]
    ax.hist(range_diffs, bins=20, edgecolor='black')
    ax.set_xlabel('距离差 (m)')
    ax.set_ylabel('频次')
    ax.set_title('距离差分布')
    ax.grid(True, alpha=0.3)
    
    # 4. 方位角差分布直方图
    ax = axes[1, 1]
    az_diffs = [m['az_diff'] for m in matches]
    ax.hist(az_diffs, bins=20, edgecolor='black')
    ax.set_xlabel('方位角差 (度)')
    ax.set_ylabel('频次')
    ax.set_title('方位角差分布')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('track_comparison.png', dpi=150)
    plt.show()
    print("已保存图片: track_comparison.png")

def plot_track_trajectory(raw_df, mht_tracks, track_id=None):
    """绘制单个航迹的轨迹对比"""
    # 如果指定了track_id，只显示该航迹
    if track_id:
        raw_track = raw_df[raw_df['track_id'] == track_id]
        mht_track = [t for t in mht_tracks if t.get('track_id') == track_id]
        
        if len(raw_track) == 0 and len(mht_track) == 0:
            print(f"未找到航迹: {track_id}")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 距离随时间变化
        ax1 = plt.subplot(1, 2, 1)
        if len(raw_track) > 0:
            ax1.plot(raw_track['timestamp'], raw_track['range'], 'ro-', label='原始', markersize=4)
        if len(mht_track) > 0:
            mht_times = [t.get('frame_timestamp', 0) for t in mht_track]
            mht_ranges = [t.get('range', 0) for t in mht_track]
            ax1.plot(mht_times, mht_ranges, 'b*-', label='MHT', markersize=6)
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('距离 (m)')
        ax1.set_title(f'航迹 {track_id} - 距离变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 方位角随时间变化
        ax2 = plt.subplot(1, 2, 2)
        if len(raw_track) > 0:
            ax2.plot(raw_track['timestamp'], raw_track['azimuth'], 'ro-', label='原始', markersize=4)
        if len(mht_track) > 0:
            mht_azimuths = [t.get('azimuth', 0) for t in mht_track]
            ax2.plot(mht_times, mht_azimuths, 'b*-', label='MHT', markersize=6)
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('方位角 (度)')
        ax2.set_title(f'航迹 {track_id} - 方位角变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'track_{track_id}_comparison.png', dpi=150)
        plt.show()
        print(f"已保存图片: track_{track_id}_comparison.png")
    else:
        # 显示所有匹配的航迹
        matched_mht_ids = set([m['mht_track_id'] for m in matches])
        for mht_id in list(matched_mht_ids)[:5]:  # 最多显示5个
            plot_track_trajectory(raw_df, mht_tracks, mht_id)

def main():
    print("="*60)
    print("雷达原始航迹 vs MHT处理航迹 对比工具")
    print("="*60)
    
    # 加载数据
    raw_df = load_raw_tracks('raw_tracks.csv')
    mht_tracks = load_mht_tracks('track_results.json')
    
    if raw_df is None or len(mht_tracks) == 0:
        print("数据加载失败，请确保:")
        print("  1. raw_tracks.csv 存在且有数据")
        print("  2. track_results.json 存在且有数据")
        return
    
    # 匹配航迹
    print("\n正在匹配航迹...")
    matches = match_tracks(raw_df, mht_tracks)
    
    # 打印统计
    print_comparison_stats(matches, raw_df, mht_tracks)
    
    # 绘制对比图
    if len(matches) > 0:
        print("\n正在生成对比图...")
        plot_comparison(matches, raw_df, mht_tracks)
        
        # 询问是否查看单个航迹
        show_track = input("\n是否查看单个航迹的详细对比? (y/n): ").strip().lower()
        if show_track == 'y':
            track_id = input("请输入MHT航迹ID (如 Radar-57): ").strip()
            plot_track_trajectory(raw_df, mht_tracks, track_id)
    else:
        print("\n没有找到匹配的航迹，请检查:")
        print("  1. 时间戳是否对齐")
        print("  2. 距离和方位角范围是否一致")
        print("  3. 原始数据中是否有 range, azimuth 字段")

if __name__ == "__main__":
    main()