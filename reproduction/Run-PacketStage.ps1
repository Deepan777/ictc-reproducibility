param(
    [Parameter(Mandatory=$true)][ValidateSet('development','fit','validation','select','test','analyze')][string]$Stage,
    [Parameter(Mandatory=$true)][string]$WorkRoot,
    [Parameter(Mandatory=$true)][ValidateSet('abilene','geant-2012-03','garr-2011-09','geant-stress')][string]$Topology,
    [Parameter(Mandatory=$true)][string]$Python,
    [string]$Ns3LibraryBin,
    [string]$CompilerRuntimeBin,
    [int]$Workers=4
)
$ErrorActionPreference='Stop'
$source=Split-Path -Parent $PSScriptRoot
$env:ICTC_REPRO_ROOT=(Resolve-Path -LiteralPath $WorkRoot).Path
$env:PYTHONPATH=(Join-Path $source 'code/src')+';'+(Join-Path $source 'code')
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
if ($Ns3LibraryBin) { $env:ICTC_NS3_LIBRARY_BIN=(Resolve-Path -LiteralPath $Ns3LibraryBin).Path }
if ($CompilerRuntimeBin) { $env:ICTC_RUNTIME_BIN=(Resolve-Path -LiteralPath $CompilerRuntimeBin).Path }
if ($Stage -in @('development','validation','test')) {
    & $Python (Join-Path $source 'code/run_revision_native.py') run --topology $Topology --phase $Stage --workers $Workers
} elseif ($Stage -in @('fit','select')) {
    & $Python (Join-Path $source 'code/fit_revision_models.py') $Stage --topology $Topology
} else {
    & $Python (Join-Path $source 'code/analyze_revision.py') revision --topology $Topology
}
if ($LASTEXITCODE -ne 0) { throw "Packet stage failed: $Stage / $Topology" }
