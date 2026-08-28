#Requires -Version 7.0
$ErrorActionPreference = 'Continue'
$root = 'd:\docker\gigaam'
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$baseUrl = 'http://127.0.0.1:5002'
$hostIp = '192.168.148.109'
$mp3Source = 'd:\docker\test.mp3'
$wavChunk = Join-Path $root 'test-chunk-16k.wav'
$results = [ordered]@{}
$notes = [System.Collections.Generic.List[string]]::new()
$wavJson = $null

function Add-Result([string]$Name, [bool]$Pass, [string]$Detail) {
    $results[$Name] = [ordered]@{ pass = $Pass; detail = $Detail }
}

Write-Host "=== GigaAM ASR acceptance (pwsh $($PSVersionTable.PSVersion)) ==="

$smi = (nvidia-smi 2>&1 | Out-String)
Add-Result 'nvidia-smi GTX 1080' ($smi -match 'GTX 1080') (($smi -split "`n" | Select-Object -Skip 7 -First 3) -join ' | ')

Push-Location $root
$vram = (docker compose exec -T gigaam-asr nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 | Out-String).Trim()
Pop-Location
Add-Result 'VRAM GigaAM loaded' ($vram -match '\d') $vram

$health = (& curl.exe -s --max-time 30 "$baseUrl/health")
Add-Result 'GET /health localhost' ($health -match '"status"\s*:\s*"ok"') $health

$healthLan = (& curl.exe -s --max-time 30 "http://${hostIp}:5002/health")
Add-Result "GET /health ${hostIp}" ($healthLan -match '"status"\s*:\s*"ok"') $healthLan

if (-not (Test-Path $mp3Source)) {
    Add-Result 'test.mp3 source' $false "Not found: $mp3Source"
} else {
    $dur = (ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $mp3Source 2>&1 | Out-String).Trim()
    Add-Result 'test.mp3 source' $true "duration_sec=$dur path=$mp3Source"
    if (-not (Test-Path $wavChunk)) {
        ffmpeg -y -hide_banner -loglevel error -i $mp3Source -ss 120 -t 4 -ar 16000 -ac 1 $wavChunk
    }
}

$wavResp = (& curl.exe -s --max-time 180 -X POST "$baseUrl/api/v1/transcribe-chunk" -F "file=@$wavChunk")
try {
    $wavJson = $wavResp | ConvertFrom-Json
    $pass = ($wavJson.text.Length -gt 0) -and ([double]$wavJson.processing_time -lt 10) -and ($wavJson.language -eq 'ru')
    Add-Result 'POST transcribe-chunk WAV 4s ru' $pass $wavResp
} catch {
    Add-Result 'POST transcribe-chunk WAV 4s ru' $false $wavResp
}

$empty = Join-Path $logs 'empty.wav'
[IO.File]::WriteAllBytes($empty, [byte[]]@())
$emptyCode = (& curl.exe -s --max-time 10 -o NUL -w '%{http_code}' -X POST "$baseUrl/api/v1/transcribe-chunk" -F "file=@$empty")
Add-Result 'POST empty file -> 400' ($emptyCode -eq '400') "http_code=$emptyCode"

$fw = (netsh advfirewall firewall show rule name="GigaAM ASR 5002" 2>&1 | Out-String)
if ($fw -notmatch 'Enabled:\s+Yes') {
    $fw = (netsh advfirewall firewall show rule name="Whisper ASR 5002" 2>&1 | Out-String)
    $notes.Add('Using existing Whisper ASR 5002 firewall rule if GigaAM rule missing')
}
Add-Result 'Firewall 5002 from 192.168.149.0/24' ($fw -match 'Enabled:\s+Yes' -and $fw -match 'LocalPort:\s+5002') 'TCP 5002 allow from 192.168.149.0/24'

$tcp5002 = (Test-NetConnection -ComputerName $hostIp -Port 5002 -WarningAction SilentlyContinue).TcpTestSucceeded
Add-Result "ASR ${hostIp}:5002 TCP" $tcp5002 "TcpTestSucceeded=$tcp5002"
$notes.Add('Verify from Linux: curl -s http://192.168.148.109:5002/health')

$passed = @($results.Values | Where-Object { $_.pass }).Count
$total = $results.Count
$now = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$textOut = if ($wavJson) { $wavJson.text } else { 'n/a' }
$procTime = if ($wavJson) { $wavJson.processing_time } else { 'n/a' }

$report = @"
# GigaAM ASR — результат приёмки (TZ §6)

**Дата:** $now  
**Хост:** Windows Server 2022, $hostIp  
**URL ASR:** http://${hostIp}:5002  
**Модель:** ai-sage/GigaAM-v3 e2e_rnnt (`d:\docker\gigaam\model`)  
**Контейнер:** gigaam-asr  

---

## Чеклист

| # | Пункт | Статус | Детали |
|---|-------|--------|--------|
"@

$i = 1
foreach ($key in $results.Keys) {
    $r = $results[$key]
    $st = if ($r.pass) { 'PASS' } else { 'FAIL' }
    $d = ($r.detail -replace '\|', '/' -replace "`r?`n", ' ').Trim()
    if ($d.Length -gt 120) { $d = $d.Substring(0, 117) + '...' }
    $report += "| $i | $key | **$st** | $d |`n"
    $i++
}

$report += @"

---

## Итог: $passed / $total

- Чанк: `test-chunk-16k.wav` (4 с, 16 kHz mono)
- Текст: «$textOut»
- processing_time: $procTime с

### Linux gateway

``````env
ASR_BASE_URL=http://192.168.148.109:5002
APP_MODE=voice_bot
``````

"@

$outFile = Join-Path $root 'ACCEPTANCE-RESULT.md'
[System.IO.File]::WriteAllText($outFile, $report, [System.Text.UTF8Encoding]::new($true))
Write-Host "Saved: $outFile"
Write-Host "Passed: $passed / $total"
