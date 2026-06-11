param(
    [string]$InterfaceAlias = "",
    [string]$IpAddress = "192.168.0.9",
    [int]$PrefixLength = 24,
    [switch]$Apply
)

$ErrorActionPreference = "Stop"

function Show-Plan {
    param(
        [string]$Alias,
        [string]$Ip,
        [int]$Prefix
    )

    Write-Host "=== Optical Static IP Plan ==="
    Write-Host "interface : $Alias"
    Write-Host "ip        : $Ip"
    Write-Host "prefix    : $Prefix"
    Write-Host "subnet    : 255.255.255.0 (when prefix=24)"
    Write-Host
    Write-Host "Preview only. No network change yet."
    Write-Host "To really apply, run again with -Apply."
}

function Require-Admin {
    $current = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $current.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "This script needs Administrator PowerShell when using -Apply."
    }
}

if ([string]::IsNullOrWhiteSpace($InterfaceAlias)) {
    Write-Host "Available IPv4 interfaces:"
    Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -notlike "169.254.*" -and $_.IPAddress -ne "127.0.0.1" } |
        Select-Object InterfaceAlias, IPAddress, PrefixLength |
        Format-Table -AutoSize
    Write-Host
    Write-Host "Example:"
    Write-Host ".\\set_optical_network_static_ip.ps1 -InterfaceAlias 'Ethernet' -IpAddress 192.168.0.9"
    exit 0
}

Show-Plan -Alias $InterfaceAlias -Ip $IpAddress -Prefix $PrefixLength

if (-not $Apply) {
    exit 0
}

Require-Admin

$adapter = Get-NetAdapter -Name $InterfaceAlias -ErrorAction Stop
if (-not $adapter) {
    throw "Interface '$InterfaceAlias' not found."
}

Write-Host
Write-Host "Applying static IPv4 config..."

Get-NetIPAddress -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    ForEach-Object {
        Remove-NetIPAddress -InputObject $_ -Confirm:$false -ErrorAction SilentlyContinue
    }

Get-NetRoute -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.DestinationPrefix -eq "0.0.0.0/0" } |
    ForEach-Object {
        Remove-NetRoute -InputObject $_ -Confirm:$false -ErrorAction SilentlyContinue
    }

New-NetIPAddress -InterfaceAlias $InterfaceAlias -IPAddress $IpAddress -PrefixLength $PrefixLength -AddressFamily IPv4 | Out-Null

Write-Host "Done."
Get-NetIPAddress -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 |
    Select-Object InterfaceAlias, IPAddress, PrefixLength |
    Format-Table -AutoSize
