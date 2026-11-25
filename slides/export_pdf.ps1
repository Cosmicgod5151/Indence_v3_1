# Export the deck to PDF with PowerPoint (for visual QA).
param([string]$Pptx = "$PSScriptRoot\..\Indence_Final_Presentation.pptx")
$Pptx = (Resolve-Path $Pptx).Path
$Pdf = [System.IO.Path]::ChangeExtension($Pptx, ".pdf")
$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open($Pptx, $true, $false, $false)  # read-only, no window
    $pres.SaveAs($Pdf, 32)  # ppSaveAsPDF
    $pres.Close()
} finally {
    $app.Quit()
}
Write-Output "PDF $Pdf"
