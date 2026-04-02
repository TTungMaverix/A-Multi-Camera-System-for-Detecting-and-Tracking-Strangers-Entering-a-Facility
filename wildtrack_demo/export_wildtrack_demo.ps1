param(
    [string]$ConfigPath = ""
)

Add-Type -AssemblyName System.Drawing

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $scriptRoot "wildtrack_demo_config.json"
}
$config = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$datasetRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot $config.dataset_root))
$outputRoot = Join-Path $scriptRoot "output"
$tracksDir = Join-Path $outputRoot "tracks"
$eventsDir = Join-Path $outputRoot "events"
$cropsDir = Join-Path $outputRoot "best_shots"
$masksDir = Join-Path $outputRoot "masks"
$overlaysDir = Join-Path $outputRoot "overlays"
$summaryDir = Join-Path $outputRoot "summary"
$dirs = @($outputRoot, $tracksDir, $eventsDir, $cropsDir, $masksDir, $overlaysDir, $summaryDir)
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

function Test-PointInPolygon([double]$x, [double]$y, $polygon) {
    $inside = $false
    $j = $polygon.Count - 1
    for ($i = 0; $i -lt $polygon.Count; $i++) {
        $xi = [double]$polygon[$i][0]
        $yi = [double]$polygon[$i][1]
        $xj = [double]$polygon[$j][0]
        $yj = [double]$polygon[$j][1]
        $dy = $yj - $yi
        if ([math]::Abs($dy) -lt 1e-9) { $dy = 1e-9 }
        $intersect = (($yi -gt $y) -ne ($yj -gt $y)) -and ($x -lt (($xj - $xi) * ($y - $yi) / $dy + $xi))
        if ($intersect) { $inside = -not $inside }
        $j = $i
    }
    return $inside
}

function Get-LineCrossValue($point, $line) {
    $x1 = [double]$line[0][0]
    $y1 = [double]$line[0][1]
    $x2 = [double]$line[1][0]
    $y2 = [double]$line[1][1]
    return (($x2 - $x1) * ($point.y - $y1)) - (($y2 - $y1) * ($point.x - $x1))
}

function Test-IsInSide($point, $line, $inSidePoint) {
    $anchor = @{ x = [double]$inSidePoint[0]; y = [double]$inSidePoint[1] }
    $anchorSign = Get-LineCrossValue $anchor $line
    $pointSign = Get-LineCrossValue $point $line
    return ($anchorSign * $pointSign) -ge 0
}

function Get-PointToSegmentDistance($point, $line) {
    $x1 = [double]$line[0][0]
    $y1 = [double]$line[0][1]
    $x2 = [double]$line[1][0]
    $y2 = [double]$line[1][1]
    $px = [double]$point.x
    $py = [double]$point.y
    $dx = $x2 - $x1
    $dy = $y2 - $y1
    $lengthSq = $dx * $dx + $dy * $dy
    if ($lengthSq -le 1e-9) {
        return [math]::Sqrt((($px - $x1) * ($px - $x1)) + (($py - $y1) * ($py - $y1)))
    }
    $t = ((($px - $x1) * $dx) + (($py - $y1) * $dy)) / $lengthSq
    if ($t -lt 0) { $t = 0 }
    if ($t -gt 1) { $t = 1 }
    $projX = $x1 + ($t * $dx)
    $projY = $y1 + ($t * $dy)
    return [math]::Sqrt((($px - $projX) * ($px - $projX)) + (($py - $projY) * ($py - $projY)))
}

function Test-EntryCrossing($prevPoint, $currPoint, $line, $inSidePoint, [double]$distanceThreshold) {
    $prevIn = Test-IsInSide $prevPoint $line $inSidePoint
    $currIn = Test-IsInSide $currPoint $line $inSidePoint
    if ($prevIn -or (-not $currIn)) {
        return $false
    }
    $midPoint = @{ x = (($prevPoint.x + $currPoint.x) / 2.0); y = (($prevPoint.y + $currPoint.y) / 2.0) }
    $distance = Get-PointToSegmentDistance $midPoint $line
    return $distance -le $distanceThreshold
}

function Export-CropImage($imagePath, $rect, $outPath) {
    $img = [System.Drawing.Image]::FromFile($imagePath)
    $bmp = New-Object System.Drawing.Bitmap ([int]$rect.width), ([int]$rect.height)
    $gfx = [System.Drawing.Graphics]::FromImage($bmp)
    $srcRect = New-Object System.Drawing.Rectangle ([int]$rect.x, [int]$rect.y, [int]$rect.width, [int]$rect.height)
    $dstRect = New-Object System.Drawing.Rectangle 0, 0, ([int]$rect.width), ([int]$rect.height)
    $gfx.DrawImage($img, $dstRect, $srcRect, [System.Drawing.GraphicsUnit]::Pixel)
    $bmp.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $gfx.Dispose()
    $bmp.Dispose()
    $img.Dispose()
}

function Clamp-Rect($rect, [int]$imgWidth, [int]$imgHeight) {
    $x = [math]::Max(0, [int]$rect.x)
    $y = [math]::Max(0, [int]$rect.y)
    $maxW = $imgWidth - $x
    $maxH = $imgHeight - $y
    $w = [math]::Min([int]$rect.width, $maxW)
    $h = [math]::Min([int]$rect.height, $maxH)
    return @{ x = $x; y = $y; width = [math]::Max(1, $w); height = [math]::Max(1, $h) }
}

function Get-HeuristicHeadRect($record, $headCfg, [int]$imgWidth, [int]$imgHeight) {
    $w = [double]$record.width
    $h = [double]$record.height
    $x = [double]$record.xmin + ($headCfg.side_ratio * $w)
    $y = [double]$record.ymin + ($headCfg.top_ratio * $h)
    $headW = $w * (1.0 - (2.0 * $headCfg.side_ratio))
    $headH = $h * ($headCfg.bottom_ratio - $headCfg.top_ratio)
    return Clamp-Rect @{ x = [int][math]::Round($x); y = [int][math]::Round($y); width = [int][math]::Round($headW); height = [int][math]::Round($headH) } $imgWidth $imgHeight
}

function Save-MaskAndOverlay($cameraName, $cameraCfg, $datasetRoot, $masksDir, $overlaysDir) {
    $sampleImagePath = Join-Path $datasetRoot $cameraCfg.sample_image
    $image = [System.Drawing.Image]::FromFile($sampleImagePath)
    $width = $image.Width
    $height = $image.Height
    $bitmap = New-Object System.Drawing.Bitmap $image
    $image.Dispose()

    $mask = New-Object System.Drawing.Bitmap $width, $height
    $maskGraphics = [System.Drawing.Graphics]::FromImage($mask)
    $maskGraphics.Clear([System.Drawing.Color]::Black)
    $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::White)
    $polygon = if ($cameraCfg.role -eq 'entry') { $cameraCfg.entry_roi } else { $cameraCfg.track_roi }
    $points = @()
    foreach ($pt in $polygon) {
        $points += New-Object System.Drawing.Point([int]$pt[0], [int]$pt[1])
    }
    $maskGraphics.FillPolygon($brush, $points)
    $maskGraphics.Dispose()
    $maskPath = Join-Path $masksDir ($cameraName + '_mask.png')
    $mask.Save($maskPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $mask.Dispose()

    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $roiPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(230, 255, 220, 0)), 6
    $trackPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(230, 0, 255, 255)), 6
    $cropPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(230, 255, 64, 64)), 5
    $linePen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(230, 0, 255, 0)), 6
    $font = New-Object System.Drawing.Font('Arial', 28, [System.Drawing.FontStyle]::Bold)
    $textBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(235, 255, 255, 255))

    if ($cameraCfg.role -eq 'entry') {
        $graphics.DrawPolygon($roiPen, $points)
        $p1 = New-Object System.Drawing.Point([int]$cameraCfg.entry_line[0][0], [int]$cameraCfg.entry_line[0][1])
        $p2 = New-Object System.Drawing.Point([int]$cameraCfg.entry_line[1][0], [int]$cameraCfg.entry_line[1][1])
        $graphics.DrawLine($linePen, $p1, $p2)
        $graphics.DrawString('IN', $font, $textBrush, [float]$cameraCfg.in_side_point[0], [float]$cameraCfg.in_side_point[1])
    }
    else {
        $graphics.DrawPolygon($trackPen, $points)
    }

    if ($cameraCfg.crop) {
        $crop = $cameraCfg.crop
        $graphics.DrawRectangle($cropPen, [int]$crop.x, [int]$crop.y, [int]$crop.width, [int]$crop.height)
    }

    $graphics.DrawString($cameraName, $font, $textBrush, 30.0, 24.0)
    $overlayPath = Join-Path $overlaysDir ($cameraName + '_overlay.png')
    $bitmap.Save($overlayPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $bitmap.Dispose()
}

$fps = [double]$config.assumed_video_fps
$bestShotWindow = [int]$config.best_shot_window_frames
$lineThreshold = [double]$config.line_crossing_distance_threshold
$annotationDir = Join-Path $datasetRoot 'annotations_positions'
$annotationFiles = Get-ChildItem -File -LiteralPath $annotationDir | Sort-Object Name

$cameraTrackMap = @{}
$cameraRowsMap = @{}
foreach ($cameraName in $config.selected_cameras) {
    $cameraTrackMap[$cameraName] = @{}
    $cameraRowsMap[$cameraName] = New-Object System.Collections.Generic.List[object]
}

foreach ($cameraName in $config.selected_cameras) {
    Save-MaskAndOverlay $cameraName $config.cameras.$cameraName $datasetRoot $masksDir $overlaysDir
}

foreach ($file in $annotationFiles) {
    $frameId = [int][System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    $frameKey = '{0:D8}' -f $frameId
    $relativeSec = [math]::Round(($frameId / $fps), 3)
    $objects = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($cameraName in $config.selected_cameras) {
        $cameraCfg = $config.cameras.$cameraName
        $polygon = if ($cameraCfg.role -eq 'entry') { $cameraCfg.entry_roi } else { $cameraCfg.track_roi }
        foreach ($obj in $objects) {
            $view = $obj.views[[int]$cameraCfg.view_index]
            if ($view.xmin -lt 0 -or $view.ymin -lt 0 -or $view.xmax -le $view.xmin -or $view.ymax -le $view.ymin) {
                continue
            }
            $width = [int]($view.xmax - $view.xmin)
            $height = [int]($view.ymax - $view.ymin)
            $centerX = [math]::Round((($view.xmin + $view.xmax) / 2.0), 2)
            $centerY = [math]::Round((($view.ymin + $view.ymax) / 2.0), 2)
            $footX = $centerX
            $footY = [double]$view.ymax
            if (-not (Test-PointInPolygon $footX $footY $polygon)) {
                continue
            }
            $row = [PSCustomObject]@{
                camera_id = $cameraName
                role = $cameraCfg.role
                global_gt_id = [int]$obj.personID
                position_id = [int]$obj.positionID
                frame_id = $frameId
                frame_key = $frameKey
                relative_sec = $relativeSec
                xmin = [int]$view.xmin
                ymin = [int]$view.ymin
                xmax = [int]$view.xmax
                ymax = [int]$view.ymax
                width = $width
                height = $height
                area = $width * $height
                center_x = $centerX
                center_y = $centerY
                foot_x = $footX
                foot_y = $footY
                image_rel_path = ('Image_subsets\\{0}\\{1}.png' -f $cameraName, $frameKey)
            }
            $cameraRowsMap[$cameraName].Add($row)
            if (-not $cameraTrackMap[$cameraName].ContainsKey([string]$obj.personID)) {
                $cameraTrackMap[$cameraName][[string]$obj.personID] = New-Object System.Collections.Generic.List[object]
            }
            $cameraTrackMap[$cameraName][[string]$obj.personID].Add($row)
        }
    }
}

$allRows = New-Object System.Collections.Generic.List[object]
foreach ($cameraName in $config.selected_cameras) {
    $cameraRows = $cameraRowsMap[$cameraName] | Sort-Object frame_id, global_gt_id
    foreach ($row in $cameraRows) { $allRows.Add($row) }
    $cameraRows | Export-Csv -LiteralPath (Join-Path $tracksDir ($cameraName + '_tracks.csv')) -NoTypeInformation -Encoding UTF8
}
$allRows | Export-Csv -LiteralPath (Join-Path $tracksDir 'all_tracks_filtered.csv') -NoTypeInformation -Encoding UTF8

$events = New-Object System.Collections.Generic.List[object]
$eventCsvRows = New-Object System.Collections.Generic.List[object]
$identityQueueRows = New-Object System.Collections.Generic.List[object]
$globalSummaryMap = @{}
$eventCounter = 0

foreach ($cameraName in $config.selected_cameras) {
    $cameraRows = $cameraRowsMap[$cameraName]
    foreach ($row in $cameraRows) {
        if (-not $globalSummaryMap.ContainsKey([string]$row.global_gt_id)) {
            $globalSummaryMap[[string]$row.global_gt_id] = [ordered]@{
                global_gt_id = $row.global_gt_id
                cameras_seen = New-Object System.Collections.Generic.HashSet[string]
                first_frame = $row.frame_id
                last_frame = $row.frame_id
                first_sec = $row.relative_sec
                last_sec = $row.relative_sec
            }
        }
        $item = $globalSummaryMap[[string]$row.global_gt_id]
        [void]$item.cameras_seen.Add($cameraName)
        if ($row.frame_id -lt $item.first_frame) { $item.first_frame = $row.frame_id; $item.first_sec = $row.relative_sec }
        if ($row.frame_id -gt $item.last_frame) { $item.last_frame = $row.frame_id; $item.last_sec = $row.relative_sec }
    }
}

foreach ($cameraName in $config.selected_cameras) {
    $cameraCfg = $config.cameras.$cameraName
    if ($cameraCfg.role -ne 'entry') { continue }
    foreach ($trackId in $cameraTrackMap[$cameraName].Keys) {
        $records = $cameraTrackMap[$cameraName][$trackId] | Sort-Object frame_id
        if ($records.Count -lt 2) { continue }
        for ($i = 1; $i -lt $records.Count; $i++) {
            $prev = $records[$i - 1]
            $curr = $records[$i]
            $prevPoint = @{ x = [double]$prev.foot_x; y = [double]$prev.foot_y }
            $currPoint = @{ x = [double]$curr.foot_x; y = [double]$curr.foot_y }
            if (-not (Test-EntryCrossing $prevPoint $currPoint $cameraCfg.entry_line $cameraCfg.in_side_point $lineThreshold)) {
                continue
            }

            $windowStart = [math]::Max([int]$curr.frame_id - 20, 0)
            $windowEnd = [int]$curr.frame_id + $bestShotWindow
            $candidates = $records | Where-Object { $_.frame_id -ge $windowStart -and $_.frame_id -le $windowEnd } | Sort-Object area -Descending
            if (-not $candidates -or $candidates.Count -eq 0) {
                $candidates = $records | Sort-Object area -Descending
            }
            $best = $candidates | Select-Object -First 1
            $imagePath = Join-Path $datasetRoot $best.image_rel_path
            $eventCounter++
            $eventId = ('IN_{0}_{1}_{2:D8}' -f $cameraName, $trackId, $curr.frame_id)

            $bodyDir = Join-Path $cropsDir $cameraName
            New-Item -ItemType Directory -Force -Path $bodyDir | Out-Null
            $bodyPath = Join-Path $bodyDir ($eventId + '_body.png')
            $headPath = Join-Path $bodyDir ($eventId + '_head.png')

            $img = [System.Drawing.Image]::FromFile($imagePath)
            $bodyRect = Clamp-Rect @{ x = $best.xmin; y = $best.ymin; width = $best.width; height = $best.height } $img.Width $img.Height
            $headRect = Get-HeuristicHeadRect $best $config.head_crop $img.Width $img.Height
            $img.Dispose()
            Export-CropImage $imagePath $bodyRect $bodyPath
            Export-CropImage $imagePath $headRect $headPath

            $event = [ordered]@{
                event_id = $eventId
                event_type = 'ENTRY_IN'
                camera_id = $cameraName
                global_gt_id = [int]$trackId
                frame_id = [int]$curr.frame_id
                relative_sec = [double]$curr.relative_sec
                identity_status = 'pending'
                resolved_identity = ''
                unknown_global_id = ''
                status_stub = 'resolve_with_face_matching_then_known_unknown'
                best_shot_frame = [int]$best.frame_id
                best_body_crop = $bodyPath
                best_head_crop = $headPath
                source_image = $imagePath
                bbox = [ordered]@{ xmin = $best.xmin; ymin = $best.ymin; xmax = $best.xmax; ymax = $best.ymax }
            }
            $events.Add([PSCustomObject]$event)
            $eventCsvRows.Add([PSCustomObject]@{
                event_id = $eventId
                event_type = 'ENTRY_IN'
                camera_id = $cameraName
                global_gt_id = [int]$trackId
                frame_id = [int]$curr.frame_id
                relative_sec = [double]$curr.relative_sec
                best_shot_frame = [int]$best.frame_id
                best_body_crop = $bodyPath
                best_head_crop = $headPath
                identity_status = 'pending'
                resolved_identity = ''
                unknown_global_id = ''
                status_stub = 'resolve_with_face_matching_then_known_unknown'
            })
            $identityQueueRows.Add([PSCustomObject]@{
                event_id = $eventId
                queue_stage = 'face_matching'
                camera_id = $cameraName
                global_gt_id = [int]$trackId
                frame_id = [int]$curr.frame_id
                relative_sec = [double]$curr.relative_sec
                best_head_crop = $headPath
                best_body_crop = $bodyPath
                identity_status = 'pending'
                matched_known_id = ''
                matched_known_score = ''
                unknown_global_id = ''
                next_step = 'match_face_then_assign_known_or_unknown'
            })
            break
        }
    }
}

$events | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $eventsDir 'entry_in_events.json') -Encoding UTF8
$eventCsvRows | Export-Csv -LiteralPath (Join-Path $eventsDir 'entry_in_events.csv') -NoTypeInformation -Encoding UTF8
$identityQueueRows | Export-Csv -LiteralPath (Join-Path $eventsDir 'identity_resolution_queue.csv') -NoTypeInformation -Encoding UTF8
$identityQueueRows | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $eventsDir 'identity_resolution_queue.json') -Encoding UTF8

$globalSummaries = @()
foreach ($globalId in ($globalSummaryMap.Keys | Sort-Object {[int]$_})) {
    $item = $globalSummaryMap[$globalId]
    $globalSummaries += [PSCustomObject]@{
        global_gt_id = [int]$item.global_gt_id
        cameras_seen = (($item.cameras_seen | Sort-Object) -join ',')
        first_frame = [int]$item.first_frame
        last_frame = [int]$item.last_frame
        first_sec = [double]$item.first_sec
        last_sec = [double]$item.last_sec
    }
}
$globalSummaries | Export-Csv -LiteralPath (Join-Path $summaryDir 'global_gt_summary.csv') -NoTypeInformation -Encoding UTF8

$summary = [ordered]@{
    dataset_root = $datasetRoot
    selected_cameras = $config.selected_cameras
    assumed_video_fps = $fps
    filtered_track_rows = $allRows.Count
    entry_in_events = $events.Count
    identity_queue_rows = $identityQueueRows.Count
    per_camera_rows = [ordered]@{}
    notes = @(
        'This export uses Wildtrack ground-truth annotations to prepare a demo-ready subset.',
        'Use best_head_crop and best_body_crop as candidates for known/unknown face matching.',
        'Wildtrack cameras overlap strongly, so travel-time logic is represented as overlap-based frame-gap windows instead of corridor travel time.',
        'identity_resolution_queue.csv is the handoff file for known/unknown matching in the demo pipeline.'
    )
}
foreach ($cameraName in $config.selected_cameras) {
    $summary.per_camera_rows[$cameraName] = $cameraRowsMap[$cameraName].Count
}
$summary.topology = $config.camera_topology
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $summaryDir 'demo_summary.json') -Encoding UTF8
Copy-Item -LiteralPath $ConfigPath -Destination (Join-Path $summaryDir 'wildtrack_demo_config.json') -Force

Write-Output ('Dataset root: ' + $datasetRoot)
Write-Output ('Filtered track rows: ' + $allRows.Count)
Write-Output ('ENTRY_IN events: ' + $events.Count)
Write-Output ('Identity queue rows: ' + $identityQueueRows.Count)
foreach ($cameraName in $config.selected_cameras) {
    Write-Output ('- ' + $cameraName + ': ' + $cameraRowsMap[$cameraName].Count + ' rows')
}
