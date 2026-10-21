param(
    [int]$Epochs = 20,
    [int]$BatchSize = 8,
    [int]$Workers = 2,
    [int]$MaxSteps = 0,
    [int]$Channels = 96,
    [ValidateSet('tag_ddpm_rgba_v1','tag_ddpm_rgba_v2_attention')][string]$Architecture = 'tag_ddpm_rgba_v2_attention',
    [string]$OutputDir = 'checkpoints\generator_large',
    [string]$DataDir = 'namemc-skins-dataset-main',
    [string]$InitFrom = '',
    [double]$LearningRate = 0.0002
)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskScript = Join-Path $taskRoot 'skin_generator.py'
$taskRun = if ([System.IO.Path]::IsPathRooted($OutputDir)) { [System.IO.Path]::GetFullPath($OutputDir) } else { [System.IO.Path]::GetFullPath((Join-Path $taskRoot $OutputDir)) }
$taskPidFile = Join-Path $taskRun 'process.json'
New-Item -ItemType Directory -Path $taskRun -Force | Out-Null
foreach ($taskRecord in @($taskPidFile, (Join-Path $taskRun 'status.json'))) {
    if (Test-Path -LiteralPath $taskRecord) {
        $taskPrevious = Get-Content -LiteralPath $taskRecord -Raw | ConvertFrom-Json
        $taskExisting = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskPrevious.pid)" -ErrorAction SilentlyContinue
        if ($taskExisting -and $taskExisting.CommandLine -like '*skin_generator.py*train*') {
            Write-Output "Generator training already running: PID $($taskPrevious.pid)"
            exit 0
        }
    }
}
if ((Test-Path -LiteralPath (Join-Path $taskRun 'last.pt')) -and (Test-Path -LiteralPath (Join-Path $taskRun 'config.json'))) {
    $taskSaved = Get-Content -LiteralPath (Join-Path $taskRun 'config.json') -Raw | ConvertFrom-Json
    foreach ($taskSetting in @(@{Parameter='BatchSize'; Field='batch_size'}, @{Parameter='Channels'; Field='channels'}, @{Parameter='Architecture'; Field='architecture'}, @{Parameter='LearningRate'; Field='lr'})) {
        $taskSavedValue = $taskSaved.($taskSetting.Field)
        if ($taskSetting.Parameter -eq 'Architecture' -and -not $taskSavedValue) { $taskSavedValue = 'tag_ddpm_rgba_v1' }
        if ($PSBoundParameters.ContainsKey($taskSetting.Parameter)) {
            if ((Get-Variable -Name $taskSetting.Parameter -ValueOnly) -ne $taskSavedValue) { throw "Checkpoint requires -$($taskSetting.Parameter) $taskSavedValue" }
        } else { Set-Variable -Name $taskSetting.Parameter -Value $taskSavedValue }
    }
}
$taskData = if ([System.IO.Path]::IsPathRooted($DataDir)) { [System.IO.Path]::GetFullPath($DataDir) } else { [System.IO.Path]::GetFullPath((Join-Path $taskRoot $DataDir)) }
$taskInit = if ($InitFrom) { if ([System.IO.Path]::IsPathRooted($InitFrom)) { [System.IO.Path]::GetFullPath($InitFrom) } else { [System.IO.Path]::GetFullPath((Join-Path $taskRoot $InitFrom)) } } else { '' }
if (-not (Test-Path -LiteralPath (Join-Path $taskData 'data.json'))) { throw "Dataset missing: $taskData\data.json" }
if ($taskInit -and -not (Test-Path -LiteralPath $taskInit)) { throw "Initial checkpoint missing: $taskInit" }
# Reuse the verified manifest; the Python trainer checks its signature before use.
$taskManifest = Join-Path $taskRun 'dataset.json'
$taskVerifiedManifest = Join-Path $taskRoot 'checkpoints\generator\dataset.json'
if (-not (Test-Path -LiteralPath $taskManifest) -and (Test-Path -LiteralPath $taskVerifiedManifest)) {
    Copy-Item -LiteralPath $taskVerifiedManifest -Destination $taskManifest
}
if (-not (Test-Path -LiteralPath $taskPython)) { throw "Python environment missing: $taskPython" }
$taskStopFile = Join-Path $taskRun 'STOP'
if (Test-Path -LiteralPath $taskStopFile) { Remove-Item -LiteralPath $taskStopFile }
$taskArgs = @('-u', "`"$taskScript`"", 'train', '--device', 'cuda', '--epochs', "$Epochs",
              '--batch-size', "$BatchSize", '--workers', "$Workers", '--max-steps', "$MaxSteps",
              '--channels', "$Channels", '--architecture', $Architecture, '--output', "`"$taskRun`"",
              '--data', "`"$taskData`"", '--lr', $LearningRate.ToString([System.Globalization.CultureInfo]::InvariantCulture))
if ($taskInit) { $taskArgs += @('--init-from', "`"$taskInit`"") }
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $taskRun 'train.stdout.log') `
    -RedirectStandardError (Join-Path $taskRun 'train.stderr.log')
@{pid=$taskProcess.Id; started_at=(Get-Date).ToUniversalTime().ToString('o'); epochs=$Epochs; batch_size=$BatchSize; channels=$Channels; architecture=$Architecture} |
    ConvertTo-Json | Set-Content -LiteralPath $taskPidFile -Encoding utf8
Write-Output "Started generator training: PID $($taskProcess.Id)"
Write-Output "Status: $taskRun\status.json"
Write-Output "Log: $taskRun\train.stdout.log"
