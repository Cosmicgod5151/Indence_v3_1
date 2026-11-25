# Update TOC / list fields with Microsoft Word, save, export PDF, and report page positions.
param([string]$Docx = "$PSScriptRoot\..\Indence_Final_Report.docx")
$Docx = (Resolve-Path $Docx).Path
$Pdf = [System.IO.Path]::ChangeExtension($Docx, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx)
    for ($pass = 0; $pass -lt 2; $pass++) {
        $doc.Fields.Update() | Out-Null
        foreach ($t in $doc.TablesOfContents) { $t.Update() }
    }
    $doc.Save()
    $doc.ExportAsFixedFormat($Pdf, 17)   # wdExportFormatPDF
    $pages = $doc.ComputeStatistics(2)   # wdStatisticPages
    Write-Output "TOTAL_PAGES $pages"
    foreach ($p in $doc.Paragraphs) {
        $s = $p.Style.NameLocal
        if ($s -eq "Heading 1") {
            $pg = $p.Range.Information(3)          # wdActiveEndAdjustedPageNumber
            $abs = $p.Range.Information(1)         # wdActiveEndPageNumber (physical)
            Write-Output ("H1 phys={0} shown={1} {2}" -f $abs, $pg, $p.Range.Text.Trim())
        }
    }
    $doc.Close()
} finally {
    $word.Quit()
}
Write-Output "PDF $Pdf"
