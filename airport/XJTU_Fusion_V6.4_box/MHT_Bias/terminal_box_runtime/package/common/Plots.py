from turtle import color
import numpy as np
from typing import Any, Dict, List, Sequence, Tuple, Type, Union, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
label_color = ['blue', 'red','green', 'purple', 'orange', 'brown', 'olive', 'cyan', 'pink']
from cmath import isinf, isnan
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def plot_hyp(
    Trees, assignment, time_k
) -> None:
    fig = plt.figure(figsize=(15,10), dpi=100)
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    plt.suptitle(f'Time: {time_k}; Global Tree: id | label | score; Green: Miss-Det; Blue: Death')
    Trees_Len = len(Trees)
    y_dis = 10.0 / Trees_Len
    x_dis = 1.0
    x_max = x_dis * len(Trees[-1])
    # x_anno = x_max / 12.0*0.15
    x_anno = 0
    Trees_xy = []
    y_now = 10.0
    for tree in Trees:
        tree_xy = {}
        node_num = 0
        if len(tree) >= 2:
            x_dis = x_max / len(tree)
        x_begin = x_max/2.0 - len(tree)/2.0*x_dis
        for id ,node in tree.items():
            xy = [x_begin+node_num*x_dis, y_now]
            tree_xy[id] = xy
            node_num += 1
        y_now -= y_dis
        Trees_xy.append(tree_xy)
    chosen_ids = np.where(assignment>0.5)[0].tolist()
    # plot hyp trees.
    ax = plt.subplot(111)
    ax.set_ylim(-1, 11)
    annotate_size = 8
    ax.annotate('hyp id', [x_max-x_dis, 10], color='r', fontsize = annotate_size)
    ax.annotate('track label', [x_max-x_dis, 9.5], color='r', fontsize = annotate_size)
    ax.annotate('score', [x_max-x_dis, 9], color='r', fontsize = annotate_size)
    ax.annotate('obs_id', [x_max-x_dis, 8.5], color='r', fontsize = annotate_size)
    trees_len_now = 0
    for tree, tree_xy in zip(Trees, Trees_xy):
        trees_len_now += 1
        if trees_len_now == Trees_Len: # last layer
            for id, node in tree.items():
                # score = '{:.1f}'.format(node.score)
                if not isinf(node.score) and not isnan(node.score):
                    score = round(float(node.score))
                if node.hyp_type == 'detect' or node.hyp_type == 'initial':
                    color_, marker_ = 'k', 'o'
                elif node.hyp_type == 'miss-detect':
                    color_, marker_ = 'g', 'o'
                elif node.hyp_type == 'death':
                    color_, marker_ = 'b', 'o'
                if id in chosen_ids: # best hyp
                    color_, marker_ = 'r', 'p'
                ax.scatter(tree_xy[id][0], tree_xy[id][1], color=color_, marker=marker_)
                ax.annotate(f'{id}', [tree_xy[id][0]-x_anno, tree_xy[id][1]-0.5], color=color_, fontsize = annotate_size)
                ax.annotate(f'{node.label}', [tree_xy[id][0]-x_anno, tree_xy[id][1]-0.8], color=color_, fontsize = annotate_size)
                ax.annotate(f'{score}', [tree_xy[id][0]-x_anno, tree_xy[id][1]-1.1], color=color_, fontsize = annotate_size)
                if len(node.obs_id) > 0:
                    ax.annotate(f'{node.obs_id[0]}', [tree_xy[id][0]-x_anno, tree_xy[id][1]-1.4], color=color_, fontsize = annotate_size)
        else: # higher layer
            for id, node in tree.items():
                # score = '{:.1f}'.format(node.score)
                if not isinf(node.score) and not isnan(node.score):
                    score = round(float(node.score))
                if node.hyp_type == 'detect' or node.hyp_type == 'initial':
                    color_, marker_ = 'k', 'o'
                elif node.hyp_type == 'miss-detect':
                    color_, marker_ = 'g', 'o'
                elif node.hyp_type == 'death':
                    color_, marker_ = 'b', 'o'
                if id in chosen_ids: # best hyp
                    color_, marker_ = 'r', 'p'
                ax.scatter(tree_xy[id][0], tree_xy[id][1], color=color_, marker=marker_)
                ax.annotate(f'{id}', [tree_xy[id][0]-x_anno, tree_xy[id][1]-0.5], color=color_, fontsize = annotate_size)
                ax.annotate(f'{node.label}', [tree_xy[id][0]-x_anno, tree_xy[id][1]-0.8], color=color_, fontsize = annotate_size)
                ax.annotate(f'{score}', [tree_xy[id][0]-x_anno, tree_xy[id][1]-1.1], color=color_, fontsize = annotate_size)
                if len(node.obs_id) > 0:
                    ax.annotate(f'{node.obs_id[0]}', [tree_xy[id][0]-x_anno, tree_xy[id][1]-1.4], color=color_, fontsize = annotate_size)
                if len(node.children_ids) > 0:
                    for ch_id in node.children_ids :
                        if ch_id in Trees_xy[trees_len_now].keys():
                            ax.plot([tree_xy[id][0], Trees_xy[trees_len_now][ch_id][0]],\
                                [tree_xy[id][1], Trees_xy[trees_len_now][ch_id][1]], color=color_, linestyle=None)
        y_now = y_now - y_dis

    plt.show()

def create_3d_scatter_animation(data_list, output_file='animation.mp4', fps=10, dpi=100):
    """
    创建3D散点图动画并保存为视频
    参数:
    data_list (list): 包含3行n列numpy数组的列表，每个数组代表一帧的点集
    output_file (str): 输出视频文件名
    fps (int): 帧率(每秒帧数)
    dpi (int): 视频分辨率
    返回:
    无，直接保存视频文件
    """
    # 设置图形和3D坐标轴
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 初始化散点图
    scat = ax.scatter([], [], [], c='b', marker='o', alpha=0.6)
    # 自动确定坐标轴范围
    all_points = np.hstack(data_list)
    max_range = np.max(all_points) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('3D位置点动画')

    # 设置视频写入器
    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, output_file, dpi):
        for i, data in enumerate(data_list):
            # 确保数据是3行n列
            assert data.shape[0] == 3, "每个数组必须是3行n列"

            # 更新散点图数据
            scat._offsets3d = (data[0], data[1], data[2])

            # 添加帧编号文本
            frame_text = ax.text2D(0.02, 0.95, f"帧: {i+1}/{len(data_list)}", transform=ax.transAxes)

            # 写入当前帧
            writer.grab_frame()

            # 移除帧编号文本
            frame_text.remove()

    plt.close(fig)
    print(f"动画已保存为: {output_file}")


def show_3d_scatter_animation(data_list, fps=10, axis_limit=None):
    """
    在窗口中交互式播放3D散点图动画

    参数:
    data_list (list): 包含3行n列numpy数组的列表，每个数组代表一帧的点集
    fps (int): 帧率(每秒帧数)
    """
    # 设置图形和3D坐标轴
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 初始化散点图
    scat = ax.scatter([], [], [], c='b', marker='o', alpha=0.6, s=2)

    if axis_limit == None:
        # 自动确定坐标轴范围
        all_points = np.hstack(data_list)
        max_range = np.max(np.abs(all_points)) * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    else:
        ax.set_xlim([axis_limit[0], axis_limit[1]])
        ax.set_ylim([axis_limit[2], axis_limit[3]])
        ax.set_zlim([axis_limit[4], axis_limit[5]])

    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('3D位置点动画 (按空格键暂停/播放)')

    # 添加帧编号文本
    frame_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)

    # 动画状态控制
    paused = False

    def toggle_pause(event):
        nonlocal paused
        if event.key == ' ':
            paused = not paused

    fig.canvas.mpl_connect('key_press_event', toggle_pause)

    def update(frame):
        if paused:
            return scat,

        data = data_list[frame % len(data_list)] # 循环播放
        scat._offsets3d = (data[0], data[1], data[2])
        frame_text.set_text(f"帧: {frame % len(data_list) + 1}/{len(data_list)}")
        return scat, frame_text

    # 创建动画
    ani = FuncAnimation(
        fig, update,
        frames=len(data_list) * 2, # 播放两遍
        interval=1000/fps, # 毫秒
        blit=False,
        repeat=True # 循环播放
    )

    plt.tight_layout()
    plt.show()


def Multi_Targets_Plot_2D(
    obs_all: Sequence,
    Ts: Sequence = None,
    obs_labels: Optional[Sequence] = None,
    est_tracks: Optional[Sequence] = None,
    truth_tracks: Optional[Sequence] = None,
    title: str = 'Demo',
    xlabel: str = 'x(m)',
    ylabel: str = 'y(m)',
    figsize: Tuple = (10,10),
    T: float = 0.1, # s
    keeplim_bigger: bool = False,
    keeplim_exact: bool = False,
    same_aspect: bool = False,
    keeplim_hand: Optional[Sequence] = None,
    View: Optional[Sequence] = None,
    save_file: Optional[str] = None,
    sensor_pos:Optional[Sequence] = None,
) -> None:
    """
    dynamic scatter of obs_pos in 2D

    :param obs_all: a list of observation, obs_pos[i] is n*2 np.ndarray, n means n obs at this time \
                  obs_pos[i][:,0] is Rs, obs_pos[i][:,1] is Thetas
    :param Ts: interval of observations.
    :param est_tracks: a list of tracks, est_tracks[k][i] means i_th target's estimated state at k_th timestep. est_tracks[k][i] is a dict.
                       est_tracks[k][i]['x'] is a 1*2 np.ndarray.
    :param truth_tracks: a list of tracks, truth_tracks[k][i] means i_th target's truth state at k_th timestep. truth_tracks[k][i] is a dict.
                       truth_tracks[k][i]['x'] is a 1*2 np.ndarray.
    :param T: plot interval (s)
    :param keeplim_bigger: whether to keep the range of axis a little bigger than the exact size of data.
    :param keeplim_exact: whether to keep the range of axis equal to the exact size of data.
    :param same_aspect: whether to keep x y the same size.
    :param keeplim_hand: set xy limit by hand.
    :param View: set view point.
    :param save_file: if not None, save the gif.
    :param sensor_pos: exact position of sensor.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    obs_pos = []
    for k in range(len(obs_all)):
        obs_k = []
        for i in range(obs_all[k].shape[0]):
            r, theta = obs_all[k][i,0], obs_all[k][i,1]
            obs_k.append([r*np.cos(theta), r*np.sin(theta)])
        obs_k = np.array(obs_k)
        obs_pos.append(obs_k)
    obs_pos_all = np.concatenate(obs_pos, axis=0)
    x_min, y_min = np.min(obs_pos_all, axis=0)
    x_max, y_max = np.max(obs_pos_all, axis=0)
    trajectory = {}
    
    def init():
        return ax,

    def data_gen():
        for i in range(len(obs_pos)):
            frame = {}
            frame['frame_num'] = i + 1
            frame['obs_num'] = obs_pos[i].shape[0]
            frame['obs_pos'] = obs_pos[i]
            if obs_labels is not None:
                obs_label = obs_labels[i]
                label_class = np.unique(obs_label).squeeze()
                if label_class.shape == ():
                    label_num = 1
                else:
                    label_num = label_class.shape[0]
                frame['label_class_num'] = label_num
                frame['obs_label'] = obs_label
                frame['label_class'] = label_class
            if Ts is not None:
                frame['Delta_T'] = Ts[i]

            if est_tracks is not None:
                if est_tracks[i] is not None:
                    if len(est_tracks[i]) > 0:
                        frame['est_x'] = {}
                        frame['est_x_num'] = len(est_tracks[i])
                        for label, track in est_tracks[i].items():
                            frame['est_x'][label] = np.asarray(track['x']).squeeze()
                            if label not in trajectory.keys():
                                trajectory[label] = []
                            trajectory[label].append(frame['est_x'][label])
            if sensor_pos is not None:
                frame['sensor_pos'] = sensor_pos
            if truth_tracks is not None:
                if len(truth_tracks[i]) > 0:
                    frame['truth_pos'] = []
                    frame['truth_x_num'] = len(truth_tracks[i])
                    for track in truth_tracks[i]:
                        frame['truth_pos'].append(track['x'].squeeze())
            yield frame

    def update(frame):
        ax.cla()
        frame_num = frame['frame_num']
        obs_num = frame['obs_num']
        scatter_size = 20
        if 'Delta_T' not in frame.keys():
            ax.set_title(title + f' |frame:{frame_num}|', fontsize=18)
        else:
            Delta_T = '{:.3f}'.format(frame['Delta_T'])
            # ax.set_title(title + f' |frame:{frame_num}| |Delta_T:{Delta_T}|', fontsize=18)
            ax.set_title('Show' + f' Frame .{frame_num} ', fontsize=18)
        ax.set_xlabel(xlabel, fontsize=18, fontfamily='sans-serif', fontstyle='italic')
        ax.set_ylabel(ylabel, fontsize='x-large', fontstyle='oblique')
        if same_aspect:
            xy_min = np.min([x_min, y_min])
            xy_max = np.max([x_max, y_max])
            # ax.auto_scale_xy([xy_min, xy_max], [xy_min, xy_max], [xy_min, xy_max])
            ax.set_xlim(xy_min - (xy_max - xy_min) / 2.0, xy_max + (xy_max - xy_min) / 2.0)
            ax.set_ylim(xy_min - (xy_max - xy_min) / 2.0, xy_max + (xy_max - xy_min) / 2.0)
            ax.set_zlim(xy_min - (xy_max - xy_min) / 2.0, xy_max + (xy_max - xy_min) / 2.0)
        elif keeplim_exact:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        elif keeplim_bigger:
            ax.set_xlim(x_min - (x_max - x_min) / 2.0, x_max + (x_max - x_min) / 2.0)
            ax.set_ylim(y_min - (y_max - y_min) / 2.0, y_max + (y_max - y_min) / 2.0)
        elif keeplim_hand is not None:
            ax.set_xlim(keeplim_hand[0], keeplim_hand[1])
            ax.set_ylim(keeplim_hand[2], keeplim_hand[3])

        obs_pos = frame['obs_pos']
        if 'label_class_num' in frame.keys():
            label_class_num = frame['label_class_num']
            label_class = frame['label_class']
            label = frame['obs_label']
            for i in range(label_class_num):
                if label_class_num == 1:
                    label_ = -1
                else:
                    label_ = label_class[i]
                obs_index = np.array(np.where(label==label_))
                if label_ == -1:
                    ax.scatter(obs_pos[obs_index, 0], obs_pos[obs_index, 1], \
                                marker='*', color='k', label=f'observation num: {obs_num}')
                else:
                    if i == len(label_color)-1:
                        assert False, 'the color of labels is not enough'
                    ax.scatter(obs_pos[obs_index,0], obs_pos[obs_index,1],\
                                marker='*', color=label_color[i])
        else:
            ax.scatter(obs_pos[:, 0], obs_pos[:, 1], \
                                marker='*', color='k', label=f'observation num: {obs_num}')
        if 'est_x' in frame.keys():
            est_xs = frame['est_x']
            est_x_num = frame['est_x_num']
            ie = 0
            for label, est_x in est_xs.items():
                if ie==0:
                    ax.scatter(est_x[0], est_x[1], marker='o',color='b', label=f'estimated pos {est_x_num}', s=scatter_size)  # the estimated pos of target
                else:
                    ax.scatter(est_x[0], est_x[1], marker='o',color='b', s=scatter_size)  # the estimated pos of target
                ax.annotate(f'{label}', [est_x[0], est_x[1]], color='b')
                ie += 1
                tra = np.stack(trajectory[label])
                ax.plot(tra[:,0], tra[:,1], color='b')
        if 'truth_pos' in frame.keys():
            truth_poses = frame['truth_pos']
            truth_x_num = frame['truth_x_num']
            for it, truth_pos in enumerate(truth_poses):
                if it==0:
                    ax.scatter(truth_pos[0], truth_pos[1], marker='o', color='tomato', label=f'truth pos {truth_x_num}', s=scatter_size)
                else:
                    ax.scatter(truth_pos[0], truth_pos[1], marker='o', color='tomato', s=scatter_size)
        if 'sensor_pos' in frame.keys():
            ax.scatter(frame['sensor_pos'][0], frame['sensor_pos'][1], \
                marker='s', color='orange',label='sensor')
        ax.legend(fontsize=15)
        return ax,

    anim = animation.FuncAnimation(fig, update, frames=data_gen, interval=T*1000.0, init_func=init, blit=False)
    if save_file is not None:
        anim.save(save_file,fps=5,writer='imagemagick')  # pillow
    plt.show()
