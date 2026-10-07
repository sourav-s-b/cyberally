param()
$ErrorActionPreference = 'Stop'
# All worktrees share the root project's ignored tools/model directories.
$repoGitDirectory = & git -C $PSScriptRoot rev-parse --path-format=absolute --git-common-dir
if ($LASTEXITCODE -ne 0) { throw 'Run this helper from a project Git worktree.' }
$projectDirectory = Split-Path -Path $repoGitDirectory -Parent
$ollamaDirectory = Join-Path $projectDirectory '.tools/ollama'
$ollamaExecutable = Join-Path $ollamaDirectory 'runtime/ollama.exe'
if (-not (Test-Path -LiteralPath $ollamaExecutable)) {
    throw 'Download/extract the official Windows standalone Ollama runtime into root .tools/ollama/runtime first.'
}
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_MODELS = Join-Path $projectDirectory '.tools/ollama-models'
$env:OLLAMA_NO_CLOUD = '1'
try {
    $existingService = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 2
    Write-Output ('Ollama already responds on port 11434: ' + $existingService.version)
    Write-Output 'Existing service environment is unchanged; verify its model location/cloud settings.'
    exit 0
} catch {
    # A responding service is not required when starting a fresh local runtime.
}
New-Item -ItemType Directory -Force -Path $env:OLLAMA_MODELS | Out-Null
$ollamaProcess = Start-Process -FilePath $ollamaExecutable -ArgumentList 'serve' -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $ollamaDirectory 'service.stdout.log') `
    -RedirectStandardError (Join-Path $ollamaDirectory 'service.stderr.log')
$ollamaProcess.Id | Set-Content -LiteralPath (Join-Path $ollamaDirectory 'service.pid')
Write-Output ('Started project-local Ollama PID ' + $ollamaProcess.Id + '; readiness is recorded in service.stderr.log.')
