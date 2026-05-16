# -*- coding: utf-8 -*-
"""Generate a PDF report describing the current sensor calibration modeling problem."""

import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT_DIR = r"D:\desk\airr"
OUTPUT_PDF = os.path.join(ROOT_DIR, "docu", "传感器标定数学建模汇报.pdf")
FIT_FIGURE = os.path.join(ROOT_DIR, "calibration_data", "fit_calibration_curves.png")


def get_font():
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return FontProperties(fname=path)
    return FontProperties()


FONT = get_font()


def add_text_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.96
    ax.text(0.06, y, title, fontproperties=FONT, fontsize=18, va="top")
    y -= 0.05

    for item in lines:
        if item == "":
            y -= 0.018
            continue
        wrapped = textwrap.wrap(item, width=42, break_long_words=False, break_on_hyphens=False)
        for line in wrapped:
            ax.text(0.06, y, line, fontproperties=FONT, fontsize=11.5, va="top")
            y -= 0.024
            if y < 0.06:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                y = 0.96
        y -= 0.004

    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf, title, image_path):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.96, title, fontproperties=FONT, fontsize=18, va="top")
    if os.path.exists(image_path):
        image = plt.imread(image_path)
        img_ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
        img_ax.imshow(image)
        img_ax.axis("off")
    else:
        ax.text(0.06, 0.85, f"未找到图片: {image_path}", fontproperties=FONT, fontsize=12, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)

    page1 = [
        "一、问题背景",
        "系统由雷达和光电转台两类传感器组成。雷达输出目标的距离、方位角、俯仰角；光电转台输出目标的方位角和俯仰角。",
        "标定问题的目标是：当雷达给出一个目标测量值后，能够计算出光电应当转到的角度，使两类传感器对准同一目标。",
        "",
        "二、数学建模目标",
        "记雷达测量为 (theta_r, phi_r, rho_r)，其中 theta_r 为雷达方位角，phi_r 为雷达俯仰角，rho_r 为雷达距离。",
        "记光电测量为 (theta_o, phi_o)，其中 theta_o 为光电方位角，phi_o 为光电俯仰角。",
        "希望建立映射关系：",
        "  F(theta_r, phi_r, rho_r) = (theta_o, phi_o)",
        "即根据雷达测量，预测光电应指向的角度。",
        "",
        "三、标准几何模型",
        "先将雷达极坐标转成雷达坐标系下的三维位置：",
        "  x_r = rho_r * cos(phi_r) * sin(theta_r)",
        "  y_r = rho_r * cos(phi_r) * cos(theta_r)",
        "  z_r = rho_r * sin(phi_r)",
        "记 P_r = [x_r, y_r, z_r]^T。",
        "若雷达坐标系与光电坐标系之间存在固定旋转 R 和平移 T = [dx, dy, dz]^T，则：",
        "  P_o = R * P_r + T",
        "再把 P_o 转回光电极坐标：",
        "  theta_o = atan2(x_o, y_o)",
        "  phi_o = asin(z_o / ||P_o||)",
        "因此，完整标定问题本质上是估计参数集合 {R, T}，使预测角度与实际光电角度的误差最小。",
        "可写成最小二乘目标：",
        "  min sum( (theta_o_hat - theta_o)^2 + (phi_o_hat - phi_o)^2 )",
    ]

    page2 = [
        "四、当前工程中的简化模型",
        "1. 固定角度偏移模型",
        "  theta_o = theta_r + delta_theta",
        "  phi_o = phi_r + delta_phi",
        "该模型简单，但只能表达固定偏差，无法表达偏差随距离变化。",
        "",
        "2. 位置偏移模型",
        "只拟合平移参数 (dx, dy, dz)，利用雷达距离和光电角度求目标在两坐标系下的几何对应关系。",
        "该模型可以表达“同一安装偏差在不同距离上表现为不同角度差”的现象。",
        "",
        "3. 经验补偿模型",
        "不强求完整物理外参，而直接建立补偿函数：",
        "  delta_theta = f(range)",
        "  delta_phi = g(range)",
        "或进一步写成：",
        "  delta_theta = f(theta_r, range)",
        "  delta_phi = g(theta_r, range)",
        "这类模型更偏工程实用，适合局部工作区。",
        "",
        "五、当前数据建模时遇到的主要问题",
        "1. 可观测性不足",
        "当前部分样本中，雷达俯仰角在一个距离段内几乎不变，导致和竖直方向有关的参数难以稳定求解。",
        "这会让 dz、俯仰旋转误差以及耦合项出现病态或弱可观现象。",
        "",
        "2. 数据存在分段特性",
        "同一批样本中，不同距离段的变化规律不一致。尤其俯仰差在近距离段和中远距离段的分布明显不同。",
        "因此，单一全局曲线对俯仰差拟合较差，而分段拟合效果更好。",
        "",
        "3. 存在异常突变点",
        "例如前段样本中出现接近 9 度的雷达俯仰突变值，会显著拉坏全局拟合曲线。",
        "因此需要距离门限筛选、稳定段识别和异常值剔除。",
    ]

    page3 = [
        "六、当前拟合实验的结论",
        "在当前样本上，方位差和俯仰差表现不同：",
        "1. 方位差 d_az = optical_az - radar_az 随距离变化较平滑，可以较好用一条全局曲线描述。",
        "2. 俯仰差 d_pitch = optical_pitch - radar_pitch 若混入近距离突变样本，则全局曲线误差较大。",
        "3. 在设置 min_range = 900 m 后，俯仰数据显著变平滑，单条全局曲线已经具有较好的工程可用性。",
        "",
        "七、当前问题的建模结论",
        "当前更适合把问题表述为“在完整 6DoF 模型不稳定时，建立局部经验映射模型”。",
        "推荐的实际建模策略为：",
        "1. 方位方向：建立全局补偿函数 delta_theta = f(range)。",
        "2. 俯仰方向：先做距离门限筛选，再建立全局或分段补偿函数 delta_phi = g(range)。",
        "3. 在需要更高精度时，再按稳定距离段分别拟合局部模型。",
        "",
        "八、当前工程问题的本质",
        "本问题不是“完全不能标定”，而是“不能稳定地用一套全局固定几何参数解释所有样本”。",
        "因此，需要把物理几何模型与工程经验模型结合起来：",
        "  先用几何模型解释主结构，再用局部曲线补偿剩余误差。",
        "",
        "九、建议的下一步",
        "1. 对拟合脚本继续保留 min_range 参数，用于快速试验不同距离门限。",
        "2. 对拟合效果稳定的距离段，导出曲线参数，形成程序可调用的补偿函数。",
        "3. 对未来在线标定数据，自动保存完整会话快照，便于离线重算和曲线拟合对比。",
    ]

    with PdfPages(OUTPUT_PDF) as pdf:
        add_text_page(pdf, "传感器标定数学建模汇报", page1)
        add_text_page(pdf, "传感器标定数学建模汇报（续）", page2)
        add_text_page(pdf, "传感器标定数学建模汇报（续）", page3)
        add_image_page(pdf, "拟合曲线图", FIT_FIGURE)

    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
