param(
    [string]$QueueCsv = "D:\ĐỒ ÁN TỐT NGHIỆP\wildtrack_demo\output\events\identity_resolution_queue.csv",
    [string]$OutputCsv = "D:\ĐỒ ÁN TỐT NGHIỆP\insightface_demo_assets\runtime\resolved_events_template.csv"
)

if (-not (Test-Path -LiteralPath $QueueCsv)) {
    throw "Queue CSV not found: $QueueCsv"
}

$rows = Import-Csv -LiteralPath $QueueCsv
$resolved = foreach ($row in $rows) {
    [PSCustomObject]@{
        event_id = $row.event_id
        camera_id = $row.camera_id
        frame_id = $row.frame_id
        relative_sec = $row.relative_sec
        best_head_crop = $row.best_head_crop
        best_body_crop = $row.best_body_crop
        identity_status = 'pending'
        matched_known_id = ''
        matched_known_score = ''
        unknown_global_id = ''
        resolution_source = ''
        decision_reason = 'waiting_for_face_embedding_and_matching'
    }
}

$resolved | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8
Write-Output ("Resolved-event template written to: " + $OutputCsv)
