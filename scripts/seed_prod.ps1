<#
.SYNOPSIS
    Load extraction output into the production database over an SSH tunnel.

.DESCRIPTION
    Wraps the manual flow from docs/production.md: opens an SSH tunnel to the
    VPS's Postgres, runs digitex-db populate through it (which migrates first), then
    closes the tunnel. Idempotent - re-running is safe.

    Question images live only on this machine (extraction/data/ is gitignored),
    so data reaches production from here rather than through CI.

.PARAMETER Subject
    Single subject to load, e.g. biology. Omit to load every subject.

.PARAMETER VpsHost
    VPS hostname or IP. Defaults to $env:VPS_HOST.

.PARAMETER DbPassword
    Production POSTGRES_PASSWORD. Defaults to $env:PROD_DB_PASSWORD, else prompts.

.EXAMPLE
    ./scripts/seed_prod.ps1 -VpsHost 203.0.113.10 -Subject biology
#>
[CmdletBinding()]
param(
    [string]$Subject,
    [string]$VpsHost = $env:VPS_HOST,
    [string]$VpsUser = 'root',
    [int]$VpsPort = 22,
    [string]$DbUser = 'digitex',
    [string]$DbName = 'digitex',
    [string]$DbPassword = $env:PROD_DB_PASSWORD,
    # Not 5433: docker-compose.yml already binds that for the local Postgres.
    [int]$LocalPort = 15433
)

$ErrorActionPreference = 'Stop'

if (-not $VpsHost) {
    throw 'No VPS host. Pass -VpsHost or set $env:VPS_HOST.'
}

if (-not $DbPassword) {
    $secure = Read-Host -Prompt "Production POSTGRES_PASSWORD for $DbUser@$VpsHost" -AsSecureString
    $DbPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
}

function Test-PortOpen {
    param([int]$Port)
    try {
        $client = [Net.Sockets.TcpClient]::new('127.0.0.1', $Port)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

if (Test-PortOpen -Port $LocalPort) {
    throw "Local port $LocalPort is already in use - pass a free -LocalPort."
}

# uv needs the project root to resolve the digitex-db entry point.
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

$tunnel = $null
try {
    Write-Host "opening tunnel 127.0.0.1:$LocalPort -> ${VpsHost}:5432 ..." -ForegroundColor Cyan
    $tunnel = Start-Process ssh -PassThru -NoNewWindow -ArgumentList @(
        '-N',
        '-o', 'BatchMode=yes',
        '-o', 'ExitOnForwardFailure=yes',
        '-p', $VpsPort,
        '-L', "${LocalPort}:localhost:5432",
        "$VpsUser@$VpsHost"
    )

    $deadline = (Get-Date).AddSeconds(20)
    while (-not (Test-PortOpen -Port $LocalPort)) {
        if ($tunnel.HasExited) {
            throw "SSH tunnel exited with code $($tunnel.ExitCode). Check your key and -VpsHost."
        }
        if ((Get-Date) -gt $deadline) {
            throw 'SSH tunnel did not open within 20s.'
        }
        Start-Sleep -Milliseconds 500
    }

    # config.py loads .env with override=False, so this wins over any local DSN.
    $env:DATABASE_URL = "postgresql://${DbUser}:${DbPassword}@127.0.0.1:${LocalPort}/${DbName}"

    $seedArgs = @('run', 'digitex-db', 'populate')
    if ($Subject) { $seedArgs += $Subject }

    Write-Host 'seeding (migrations run first) ...' -ForegroundColor Cyan
    & uv @seedArgs
    if ($LASTEXITCODE -ne 0) {
        throw "digitex-db populate failed with exit code $LASTEXITCODE."
    }
    Write-Host 'done.' -ForegroundColor Green
} finally {
    Remove-Item Env:\DATABASE_URL -ErrorAction SilentlyContinue
    if ($tunnel -and -not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id -Force
        Write-Host 'tunnel closed.' -ForegroundColor Cyan
    }
    Pop-Location
}
