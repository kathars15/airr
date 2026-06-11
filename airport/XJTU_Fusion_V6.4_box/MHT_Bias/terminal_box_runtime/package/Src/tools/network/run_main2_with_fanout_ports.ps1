# main2.py now starts the UDP fanout and RTSP detection window by default.
# This wrapper is kept for compatibility and explicit fanout-port clarity.
$env:AIRR_ENABLE_MANAGED_UDP_FANOUT = "1"
$env:AIRR_ENABLE_MANAGED_CV_DETECTION = "1"

& D:\anaconda\python.exe D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\main2.py
