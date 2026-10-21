param(
    [ValidateSet('auto','cpu','cuda')][string]$Backend = 'auto',
    [string]$Repo = '',
    [int]$Port = 8001,
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskLauncher = $null
$taskPrefix = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($taskVersion in @('3.12','3.13','3.11','3.10')) {
        try {
            & py "-$taskVersion" -c 'import sys; assert sys.maxsize > 2**32' 2>$null
            if ($LASTEXITCODE -eq 0) { $taskLauncher = 'py'; $taskPrefix = @("-$taskVersion"); break }
        } catch { }
    }
}
if (-not $taskLauncher) {
    foreach ($taskCandidate in @('python3','python')) {
        if (Get-Command $taskCandidate -ErrorAction SilentlyContinue) {
            try {
                & $taskCandidate -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13) and sys.maxsize > 2**32' 2>$null
                if ($LASTEXITCODE -eq 0) { $taskLauncher = $taskCandidate; break }
            } catch { }
        }
    }
}
if (-not $taskLauncher) { throw 'Install 64-bit Python 3.12 from https://www.python.org/downloads/ and try again.' }
$taskArgs = @('launch.py','--backend',$Backend,'--port',"$Port")
if ($Repo) { $taskArgs += @('--repo',$Repo) }
if ($Check) { $taskArgs += '--check' }
& $taskLauncher @taskPrefix @taskArgs
exit $LASTEXITCODE
