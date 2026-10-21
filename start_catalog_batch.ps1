param(
    [int]$Count = 1000,
    [int]$Steps = 100,
    [int]$BatchSize = 4,
    [string]$RunDir = 'output\catalog-batch',
    [string]$PublishRepo = ''
)
$ErrorActionPreference = 'Stop'
$taskRun = if ([IO.Path]::IsPathRooted($RunDir)) { [IO.Path]::GetFullPath($RunDir) } else { [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $RunDir)) }
New-Item -ItemType Directory -Path $taskRun -Force | Out-Null
$taskStatus = Join-Path $taskRun 'status.json'
$taskLock = Join-Path $taskRun 'run.lock'
if (Test-Path -LiteralPath $taskLock) {
    $taskOldPid = Get-Content -LiteralPath $taskLock -Raw
    $taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$taskOldPid" -ErrorAction SilentlyContinue
    if ($taskProcess -and $taskProcess.CommandLine -like '*generate_catalog.py*') { Write-Output "Catalog generation already running: PID $taskOldPid"; exit 0 }
    Remove-Item -LiteralPath $taskLock
}
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Install the application first using start.ps1.' }
$taskArguments = @('-u', ('"' + (Join-Path $PSScriptRoot 'tools\generate_catalog.py') + '"'), '--count', $Count, '--steps', $Steps, '--batch-size', $BatchSize, '--run', ('"' + $taskRun + '"'))
if ($PublishRepo) { $taskArguments += @('--publish-repo', ('"' + $PublishRepo + '"')) }
$taskWorker = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRun 'stdout.log') -RedirectStandardError (Join-Path $taskRun 'stderr.log') -PassThru
Write-Output "Catalog generation started: PID $($taskWorker.Id)"
Write-Output "Progress: $taskStatus"
Write-Output "Log: $(Join-Path $taskRun 'stdout.log')"
