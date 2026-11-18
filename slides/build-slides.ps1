$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$slidesDir = Join-Path $root "slides"
$files = Get-ChildItem -Path $slidesDir -Filter "slide-*.html" | Sort-Object Name

$items = @()
foreach ($f in $files) {
  $raw = [System.IO.File]::ReadAllText($f.FullName)
  $json = ConvertTo-Json $raw -Compress
  $items += "  $json"
}

$js = "window.SLIDES_BUNDLE = [`r`n" + ($items -join ",`r`n") + "`r`n];`r`n"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $slidesDir "slides-bundle.js"), $js, $utf8NoBom)
Write-Output "Built slides-bundle.js with $($files.Count) slides."
