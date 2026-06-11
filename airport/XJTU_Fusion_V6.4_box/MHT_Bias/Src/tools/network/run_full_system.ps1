$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias"
$Python = "D:\anaconda\python.exe"
$MainScript = Join-Path $ProjectRoot "Src\main2.py"
$ConfigScript = Join-Path $ProjectRoot "Src\tools\network\configure_eosmsv4_for_fanout.py"
$EosRoot = "D:\desk\tra\eosmsv4yibiaodin"
$EosExe = Join-Path $EosRoot "eosmsv4.exe"

Write-Host "[full-start] stopping old eosmsv4.exe instances..."
Get-Process -Name eosmsv4 -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

Write-Host "[full-start] configuring eosmsv4 downstream ports..."
& $Python $ConfigScript

Write-Host "[full-start] starting main2.py. It will start UDP fanout and CV detection..."
$mainProcess = Start-Process -FilePath $Python -ArgumentList "`"$MainScript`"" -WorkingDirectory (Split-Path $MainScript) -PassThru

Write-Host "[full-start] waiting for fanout to own 9000/9966..."
$deadline = (Get-Date).AddSeconds(12)
$ready = $false
while ((Get-Date) -lt $deadline) {
    $ports = Get-NetUDPEndpoint -ErrorAction SilentlyContinue |
        Where-Object {
            ($_.LocalPort -eq 9000 -or $_.LocalPort -eq 9966) -and
            ($_.OwningProcess -eq $mainProcess.Id -or (Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName -eq "python")
        }
    $has9000 = $ports | Where-Object { $_.LocalPort -eq 9000 }
    $has9966 = $ports | Where-Object { $_.LocalPort -eq 9966 }
    if ($has9000 -and $has9966) {
        $ready = $true
        break
    }
    Start-Sleep -Milliseconds 500
}

if (-not $ready) {
    Write-Warning "[full-start] fanout did not claim both 9000 and 9966 within timeout. Check main2 output."
} else {
    Write-Host "[full-start] fanout ready."
}

Write-Host "[full-start] starting eosmsv4.exe..."
Start-Process -FilePath $EosExe -WorkingDirectory $EosRoot

Write-Host "[full-start] done."
Write-Host "main2 PID: $($mainProcess.Id)"
Write-Host "fanout status: $ProjectRoot\Src\data\udp_fanout_status.json"
Write-Host "check command: & $Python $ProjectRoot\Src\tools\network\check_udp_fanout_status.py"
