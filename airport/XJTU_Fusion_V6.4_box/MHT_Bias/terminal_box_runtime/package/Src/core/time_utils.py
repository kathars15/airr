# -*- coding: utf-8 -*-
import datetime

def gps_time_to_datetime(gps_ms):
    """
    将GPS时间（毫秒）转换为 datetime 对象
    GPS时间通常从 2000-01-01 00:00:00 开始计算
    """
    # GPS纪元：2000年1月1日 00:00:00 UTC
    gps_epoch = datetime.datetime(2000, 1, 1, 0, 0, 0)
    
    # 转换为秒
    gps_seconds = gps_ms / 1000.0
    
    # 计算具体时间
    dt = gps_epoch + datetime.timedelta(seconds=gps_seconds)
    
    return dt

def format_gps_time(gps_ms):
    """格式化GPS时间为易读的字符串"""
    dt = gps_time_to_datetime(gps_ms)
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # 精确到毫秒

