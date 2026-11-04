param(
    [Parameter(Mandatory = $true)][string]$DownloadDir,
    [Parameter(Mandatory = $true)][string]$ProvenanceDir,
    [int]$WaitForProcessId = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $DownloadDir -PathType Container)) {
    throw "Download directory is missing: $DownloadDir"
}
New-Item -ItemType Directory -Path $ProvenanceDir -Force | Out-Null
$python = (Get-Command python -ErrorAction Stop).Source
$kaggle = (Get-Command kaggle -ErrorAction Stop).Source
$verify = Join-Path $PSScriptRoot 'verify_eyepacs_download.py'

function Write-Status([string]$Message) {
    Write-Output ('{0:u} {1}' -f (Get-Date).ToUniversalTime(), $Message)
}

if ($WaitForProcessId -gt 0) {
    $running = Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
    if ($null -ne $running) {
        Write-Status "Waiting for existing part 001 transfer, PID $WaitForProcessId"
        Wait-Process -Id $WaitForProcessId
    }
}

foreach ($part in 1..4) {
    $name = 'train.zip.{0:000}' -f $part
    $archive = Join-Path $DownloadDir "$name.zip"
    $proof = Join-Path $ProvenanceDir ('eyepacs-train-part-{0:000}.json' -f $part)
    if (Test-Path -LiteralPath $proof) {
        $record = Get-Content -LiteralPath $proof -Raw | ConvertFrom-Json
        if ($record.member_name -ne $name -or -not (Test-Path -LiteralPath $archive)) {
            throw "Existing provenance does not match $name"
        }
        $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $record.wrapper_sha256) {
            throw "Verified archive changed after provenance was recorded: $archive"
        }
        Write-Status "Already verified $name"
        continue
    }

    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-Status "Downloading or resuming $name, attempt $attempt"
        & $kaggle competitions download diabetic-retinopathy-detection -f $name -p $DownloadDir -q
        if ($LASTEXITCODE -eq 0) {
            & $python $verify --archive $archive --part $part --out $proof
            if ($LASTEXITCODE -eq 0) {
                Write-Status "CRC and SHA-256 verified $name"
                break
            }
        }
        if ($attempt -eq 3) {
            throw "Official Kaggle download or verification failed three times for $name"
        }
        Start-Sleep -Seconds (15 * $attempt)
    }
}

Write-Status 'All four remaining EyePACS parts verified'
