# radar_ui.py
"""
雷达数据可视化UI
功能：显示雷达点云在码盘上，点击点云获取MHT跟踪目标信息
"""

import sys
import json
import math
import threading
import time
import socket
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QTextEdit, QSplitter, QGroupBox,
    QGridLayout, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPolygonF
import numpy as np

# 雷达数据格式
RADAR_DATA_PORT = 6001  # 接收雷达点云的端口
MHT_DATA_PORT = 6002    # 接收MHT跟踪结果的端口


class RadarWidget(QWidget):
    """雷达码盘显示控件"""
    
    point_clicked = pyqtSignal(dict)  # 点击点云时发射信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []  # 存储点云 [(azimuth, distance, timestamp, target_id), ...]
        self.mht_targets = []  # 存储MHT跟踪目标
        self.selected_point = None
        self.setMinimumSize(500, 500)
        
    def update_points(self, points):
        """更新点云数据"""
        self.points = points
        self.update()
    
    def update_mht_targets(self, targets):
        """更新MHT跟踪目标"""
        self.mht_targets = targets
        self.update()
    
    def paintEvent(self, event):
        """绘制码盘"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 计算绘制区域
        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) // 2 - 40
        
        # 绘制背景
        painter.setBrush(QBrush(QColor(20, 20, 40)))
        painter.drawRect(0, 0, width, height)
        
        # 绘制外圆
        painter.setPen(QPen(QColor(100, 100, 150), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # 绘制内圆（距离圈）
        for r in [0.25, 0.5, 0.75, 1.0]:
            r_radius = int(radius * r)
            painter.setPen(QPen(QColor(60, 60, 90), 1, Qt.DashLine))
            painter.drawEllipse(center_x - r_radius, center_y - r_radius, 
                               r_radius * 2, r_radius * 2)
            
            # 距离标签
            dist_label = int(500 * r) if r > 0 else 0
            painter.setPen(QPen(QColor(150, 150, 200), 1))
            painter.drawText(center_x + r_radius - 20, center_y - 5, f"{dist_label}m")
        
        # 绘制十字线
        painter.setPen(QPen(QColor(80, 80, 120), 1))
        painter.drawLine(center_x - radius, center_y, center_x + radius, center_y)
        painter.drawLine(center_x, center_y - radius, center_x, center_y + radius)
        
        # 绘制方向标签
        painter.setPen(QPen(QColor(200, 200, 255), 1))
        painter.drawText(center_x + radius - 30, center_y - 10, "0°")
        painter.drawText(center_x + 5, center_y - radius + 15, "90°")
        painter.drawText(center_x - radius + 5, center_y - 10, "180°")
        painter.drawText(center_x + 5, center_y + radius - 5, "270°")
        
        # 绘制正北标记
        painter.setPen(QPen(QColor(255, 100, 100), 2))
        painter.drawLine(center_x, center_y - radius, center_x, center_y - radius + 15)
        painter.drawText(center_x - 5, center_y - radius + 25, "N")
        
        # 绘制点云
        for point in self.points:
            azimuth = point.get('azimuth', 0)
            distance = point.get('range', 0)
            target_id = point.get('target_id', 0)
            
            # 计算绘制位置
            rad = math.radians(90 - azimuth)  # 0°在正上方
            r = min(radius * distance / 500, radius)  # 最大500米
            x = center_x + r * math.cos(rad)
            y = center_y - r * math.sin(rad)
            
            # 根据目标类型设置颜色
            if point.get('is_track', False):
                color = QColor(255, 100, 100)  # 红色 - MHT跟踪目标
                size = 6
            elif point.get('is_selected', False):
                color = QColor(255, 255, 0)    # 黄色 - 选中
                size = 8
            else:
                color = QColor(100, 200, 100)  # 绿色 - 原始点迹
                size = 4
            
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color, 1))
            painter.drawEllipse(QPointF(x, y), size, size)
            
            # 显示ID（可选）
            if distance < 100:
                painter.setPen(QPen(QColor(200, 200, 200), 1))
                painter.drawText(x + 5, y - 5, str(target_id))
        
        # 绘制MHT跟踪目标框
        for target in self.mht_targets:
            azimuth = target.get('azimuth', 0)
            distance = target.get('distance', 0)
            
            rad = math.radians(90 - azimuth)
            r = min(radius * distance / 500, radius)
            x = center_x + r * math.cos(rad)
            y = center_y - r * math.sin(rad)
            
            # 绘制跟踪框
            painter.setPen(QPen(QColor(255, 200, 50), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x - 10, y - 10, 20, 20)
            
            # 显示目标类型
            target_type = target.get('target_type', '?')
            painter.setPen(QPen(QColor(255, 200, 50), 1))
            painter.drawText(x - 15, y - 15, target_type)
    
    def mousePressEvent(self, event):
        """鼠标点击事件 - 检测是否点击到点云"""
        if event.button() != Qt.LeftButton:
            return
        
        # 计算点击位置
        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) // 2 - 40
        
        click_x = event.pos().x()
        click_y = event.pos().y()
        
        # 查找最近的点云
        closest_point = None
        min_dist = 20  # 像素阈值
        
        for point in self.points:
            azimuth = point.get('azimuth', 0)
            distance = point.get('range', 0)
            
            rad = math.radians(90 - azimuth)
            r = min(radius * distance / 500, radius)
            x = center_x + r * math.cos(rad)
            y = center_y - r * math.sin(rad)
            
            dist = math.hypot(click_x - x, click_y - y)
            if dist < min_dist:
                min_dist = dist
                closest_point = point
        
        if closest_point:
            self.selected_point = closest_point
            self.point_clicked.emit(closest_point)
            self.update()


class RadarUI(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("雷达目标跟踪系统")
        self.setGeometry(100, 100, 1400, 800)
        
        # 数据缓存
        self.radar_points = []      # 原始雷达点迹
        self.mht_targets = []       # MHT跟踪目标
        self.selected_target_info = None
        
        # 网络接收
        self.radar_sock = None
        self.mht_sock = None
        self.running = True
        
        self.init_ui()
        self.init_network()
        self.init_timer()
    
    def init_ui(self):
        """初始化UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧：雷达码盘
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 控制栏
        control_layout = QHBoxLayout()
        self.range_combo = QComboBox()
        self.range_combo.addItems(["500m", "1000m", "2000m", "5000m"])
        self.range_combo.currentTextChanged.connect(self.on_range_changed)
        control_layout.addWidget(QLabel("显示范围:"))
        control_layout.addWidget(self.range_combo)
        
        self.show_raw_check = QComboBox()
        self.show_raw_check.addItems(["显示原始点迹", "只显示跟踪目标"])
        self.show_raw_check.currentIndexChanged.connect(self.on_display_mode_changed)
        control_layout.addWidget(QLabel("显示模式:"))
        control_layout.addWidget(self.show_raw_check)
        
        control_layout.addStretch()
        left_layout.addLayout(control_layout)
        
        # 雷达码盘
        self.radar_widget = RadarWidget()
        self.radar_widget.point_clicked.connect(self.on_point_clicked)
        left_layout.addWidget(self.radar_widget)
        
        # 右侧：信息面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setMaximumWidth(400)
        
        # 目标信息分组
        info_group = QGroupBox("目标详细信息")
        info_layout = QGridLayout(info_group)
        
        self.info_labels = {}
        info_items = [
            ("目标ID:", "target_id", ""),
            ("MHT航迹ID:", "track_id", ""),
            ("目标类型:", "target_type", ""),
            ("方位角:", "azimuth", "°"),
            ("俯仰角:", "pitch", "°"),
            ("距离:", "distance", "m"),
            ("高度:", "altitude", "m"),
            ("速度:", "speed", "m/s"),
            ("融合时间:", "fusion_time", ""),
            ("原始雷达ID:", "raw_id", ""),
        ]
        
        for i, (label, key, unit) in enumerate(info_items):
            info_layout.addWidget(QLabel(label), i, 0)
            self.info_labels[key] = QLabel("--")
            info_layout.addWidget(self.info_labels[key], i, 1)
            if unit:
                info_layout.addWidget(QLabel(unit), i, 2)
        
        right_layout.addWidget(info_group)
        
        # 目标列表
        target_group = QGroupBox("当前跟踪目标列表")
        target_layout = QVBoxLayout(target_group)
        
        self.target_table = QTableWidget()
        self.target_table.setColumnCount(5)
        self.target_table.setHorizontalHeaderLabels(["航迹ID", "目标类型", "方位角", "距离", "速度"])
        self.target_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.target_table.itemClicked.connect(self.on_table_item_clicked)
        target_layout.addWidget(self.target_table)
        
        right_layout.addWidget(target_group)
        
        # 日志输出
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        right_layout.addWidget(log_group)
        
        # 添加左右布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([1000, 400])
        main_layout.addWidget(splitter)
        
        self.max_range = 500  # 默认500米
    
    def init_network(self):
        """初始化网络接收"""
        import socket
        
        # 接收雷达点云
        self.radar_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.radar_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.radar_sock.bind(("0.0.0.0", RADAR_DATA_PORT))
        self.radar_sock.settimeout(0.1)
        
        # 接收MHT结果
        self.mht_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mht_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.mht_sock.bind(("0.0.0.0", MHT_DATA_PORT))
        self.mht_sock.settimeout(0.1)
        
        # 启动接收线程
        self.radar_thread = threading.Thread(target=self.receive_radar_data, daemon=True)
        self.mht_thread = threading.Thread(target=self.receive_mht_data, daemon=True)
        self.radar_thread.start()
        self.mht_thread.start()
        
        self.log("网络初始化完成")
    
    def init_timer(self):
        """初始化定时器"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)  # 20Hz更新
    
    def receive_radar_data(self):
        """接收雷达原始点迹"""
        while self.running:
            try:
                data, addr = self.radar_sock.recvfrom(65536)
                points = json.loads(data.decode())
                self.radar_points = points
            except socket.timeout:
                continue
            except Exception as e:
                print(f"雷达数据接收错误: {e}")
    
    def receive_mht_data(self):
        """接收MHT跟踪结果"""
        while self.running:
            try:
                data, addr = self.mht_sock.recvfrom(65536)
                result = json.loads(data.decode())
                self.mht_targets = result.get('result', [])
                
                # 更新目标列表
                self.update_target_table()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"MHT数据接收错误: {e}")
    
    def update_display(self):
        """更新显示"""
        # 过滤显示范围
        points_to_show = []
        for point in self.radar_points:
            if point.get('range', 0) <= self.max_range:
                points_to_show.append(point)
        
        # 根据显示模式过滤
        if self.show_raw_check.currentIndex() == 1:  # 只显示跟踪目标
            # 将MHT目标转换为点云格式
            mht_points = []
            for target in self.mht_targets:
                mht_points.append({
                    'azimuth': target.get('azimuth', 0),
                    'range': target.get('distance', 0),
                    'target_id': target.get('track_id', ''),
                    'is_track': True
                })
            self.radar_widget.update_points(mht_points)
        else:
            self.radar_widget.update_points(points_to_show)
        
        self.radar_widget.update_mht_targets(self.mht_targets)
    
    def update_target_table(self):
        """更新目标列表表格"""
        self.target_table.setRowCount(len(self.mht_targets))
        
        for i, target in enumerate(self.mht_targets):
            self.target_table.setItem(i, 0, QTableWidgetItem(target.get('track_id', '--')))
            
            target_type = target.get('target_type', 0)
            type_name = {0: '未知', 1: '飞鸟', 2: '无人机'}.get(target_type, str(target_type))
            self.target_table.setItem(i, 1, QTableWidgetItem(type_name))
            
            self.target_table.setItem(i, 2, QTableWidgetItem(f"{target.get('azimuth', 0):.1f}"))
            self.target_table.setItem(i, 3, QTableWidgetItem(f"{target.get('distance', 0):.0f}"))
            self.target_table.setItem(i, 4, QTableWidgetItem(f"{target.get('speed', 0):.1f}"))
    
    def on_point_clicked(self, point):
        """处理点云点击事件"""
        target_id = point.get('target_id')
        azimuth = point.get('azimuth', 0)
        distance = point.get('range', 0)
        pitch = point.get('pitch', 0)
        
        self.log(f"点击目标: ID={target_id}, 方位={azimuth:.1f}°, 距离={distance:.0f}m")
        
        # 查找对应的MHT跟踪目标
        matched_target = None
        for target in self.mht_targets:
            # 通过原始雷达ID匹配
            raw_id = target.get('extra_info', {}).get('raw_display_id', '')
            if str(target_id) in str(raw_id):
                matched_target = target
                break
        
        if matched_target:
            self.display_target_info(matched_target)
            self.log(f"匹配到MHT航迹: {matched_target.get('track_id')}")
        else:
            # 显示原始点云信息
            self.display_point_info(point)
    
    def on_table_item_clicked(self, item):
        """点击表格项"""
        row = item.row()
        if row < len(self.mht_targets):
            target = self.mht_targets[row]
            self.display_target_info(target)
            
            # 在码盘上高亮显示
            self.radar_widget.selected_point = {
                'azimuth': target.get('azimuth', 0),
                'range': target.get('distance', 0),
                'is_selected': True
            }
            self.radar_widget.update()
    
    def display_target_info(self, target):
        """显示MHT目标信息"""
        self.info_labels['target_id'].setText(str(target.get('track_id', '--')))
        self.info_labels['track_id'].setText(str(target.get('track_id', '--')))
        
        target_type = target.get('target_type', 0)
        type_name = {0: '未知', 1: '飞鸟', 2: '无人机'}.get(target_type, str(target_type))
        self.info_labels['target_type'].setText(type_name)
        
        self.info_labels['azimuth'].setText(f"{target.get('azimuth', 0):.1f}")
        self.info_labels['pitch'].setText(f"{target.get('pitch', 0):.1f}")
        self.info_labels['distance'].setText(f"{target.get('distance', 0):.0f}")
        self.info_labels['altitude'].setText(f"{target.get('alt', 0):.0f}")
        self.info_labels['speed'].setText(f"{target.get('speed', 0):.1f}")
        
        fusion_time = target.get('extra_info', {}).get('fusion_time', 0)
        self.info_labels['fusion_time'].setText(time.strftime('%H:%M:%S', time.localtime(fusion_time)))
        
        raw_id = target.get('extra_info', {}).get('raw_display_id', '')
        self.info_labels['raw_id'].setText(str(raw_id))
    
    def display_point_info(self, point):
        """显示原始点云信息"""
        self.info_labels['target_id'].setText(str(point.get('target_id', '--')))
        self.info_labels['track_id'].setText("-- (原始点迹)")
        self.info_labels['target_type'].setText("--")
        self.info_labels['azimuth'].setText(f"{point.get('azimuth', 0):.1f}")
        self.info_labels['pitch'].setText(f"{point.get('pitch', 0):.1f}")
        self.info_labels['distance'].setText(f"{point.get('range', 0):.0f}")
        self.info_labels['altitude'].setText("--")
        self.info_labels['speed'].setText(f"{point.get('speed', 0):.1f}")
        self.info_labels['fusion_time'].setText("--")
        self.info_labels['raw_id'].setText(str(point.get('target_id', '--')))
    
    def on_range_changed(self, text):
        """显示范围改变"""
        self.max_range = int(text.replace('m', ''))
        self.log(f"显示范围改为 {self.max_range}m")
    
    def on_display_mode_changed(self, index):
        """显示模式改变"""
        mode = ["原始点迹", "跟踪目标"][index]
        self.log(f"显示模式改为 {mode}")
    
    def log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """关闭事件"""
        self.running = False
        if self.radar_sock:
            self.radar_sock.close()
        if self.mht_sock:
            self.mht_sock.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = RadarUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()