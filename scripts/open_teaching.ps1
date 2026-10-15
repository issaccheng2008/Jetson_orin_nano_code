param(
    [string]$RobotHost = '192.168.43.7',
    [string]$RobotUser = 'isaac',
    [int]$LocalPort = 8765,
    [string]$RemoteDirectory = '/home/isaac/robot_teaching',
    [string]$RemotePython = '/home/isaac/venvs/humanoid_policy/bin/python'
)
$ErrorActionPreference = 'Stop'
if ($RobotHost -notmatch '^[a-zA-Z0-9.-]+$' -or $RobotUser -notmatch '^[a-zA-Z0-9_-]+$' -or
    $RemoteDirectory -notmatch '^/[a-zA-Z0-9_./-]+$' -or $RemotePython -notmatch '^/[a-zA-Z0-9_./-]+$' -or
    $LocalPort -lt 1024 -or $LocalPort -gt 65535) {
    throw 'Invalid SSH host, username, path or local port.'
}
$probe = [System.Net.Sockets.TcpClient]::new()
try {
    $probe.Connect('127.0.0.1', $LocalPort)
    throw "Local port $LocalPort is already in use. Close the earlier console or choose -LocalPort 8766."
} catch [System.Net.Sockets.SocketException] {
    # Connection refused: the requested local port is available.
} finally { $probe.Dispose() }
Write-Host 'Starting teaching console. Enter your SSH password when prompted.'
Write-Host 'Keep this window open. Closing it stops the service and motor output.'
Write-Host 'The service refuses to open a serial port owned by walking/autostart.'
$browserJob = Start-Job -ArgumentList $LocalPort -ScriptBlock {
    param($BrowserPort)
    for ($attempt = 0; $attempt -lt 180; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:$BrowserPort/api/state" -UseBasicParsing -TimeoutSec 1
            if ($response.StatusCode -eq 200) {
                # This is the interactive browser the operator explicitly launches.
                Start-Process "http://127.0.0.1:$BrowserPort/"
                break
            }
        } catch { }
        Start-Sleep -Seconds 1
    }
}
try {
    $forward = "127.0.0.1:${LocalPort}:127.0.0.1:8765"
    $remote = "exec $RemotePython -B $RemoteDirectory/teaching_console.py serve --port /dev/ttyACM0 --enable-motors --data-dir /home/isaac/jetson_orin_code/records/teaching"
    # Foreground SSH keeps its password prompt visible. The -t session owns the
    # server, so terminal closure also closes it instead of leaving a daemon.
    & ssh -t -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -L $forward "${RobotUser}@${RobotHost}" $remote
    if ($LASTEXITCODE -ne 0) { Write-Host "SSH exited with code $LASTEXITCODE" }
} finally {
    Stop-Job $browserJob -ErrorAction SilentlyContinue
    Remove-Job $browserJob -Force -ErrorAction SilentlyContinue
}
