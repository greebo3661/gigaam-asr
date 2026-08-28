netsh advfirewall firewall delete rule name="GigaAM ASR 5002" 2>$null
netsh advfirewall firewall add rule name="GigaAM ASR 5002" dir=in action=allow protocol=TCP localport=5002 remoteip=192.168.149.0/255.255.255.0 profile=any enable=yes
if ($LASTEXITCODE -eq 0) { Write-Output "OK: firewall rule added" } else { Write-Output "FAIL: exit $LASTEXITCODE"; exit 1 }
