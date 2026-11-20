param([string]$ArchiveDirectory = (Join-Path $PSScriptRoot '.upstream'))

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Path $ArchiveDirectory -Force | Out-Null

function Get-CheckedFile {
    param([string]$Name, [string]$Url, [string]$Sha256)
    $file = Join-Path $ArchiveDirectory $Name
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) {
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile "$file.download"
        if ((Get-FileHash -LiteralPath "$file.download").Hash -ne $Sha256) {
            throw "SHA-256 mismatch for $Name."
        }
        Move-Item -LiteralPath "$file.download" -Destination $file
    }
    if ((Get-FileHash -LiteralPath $file).Hash -ne $Sha256) {
        throw "SHA-256 mismatch for $file."
    }
    (Resolve-Path -LiteralPath $file).ProviderPath
}

function Expand-Members {
    param([string]$Archive, [string]$Target, [string[]]$Members)
    New-Item -ItemType Directory -Path $Target -Force | Out-Null
    & tar.exe -xf $Archive -C $Target @Members
    if ($LASTEXITCODE -ne 0) { throw "Could not extract the expected files from $Archive." }
}

$gross = Get-CheckedFile 'MultipleRank1Lattice.tar.gz' `
    'https://users.math.msu.edu/users/iwenmark/Code/MultipleRank1Lattice.tar.gz' `
    'c72b5e2621d08c17525c19440f2f771c352d8ba32dc1629874c62aaf3f03a2c1'
$generator = Get-CheckedFile 'generatorp.m' `
    'https://people.cs.kuleuven.be/~dirk.nuyens/fast-cbc/generatorp.m' `
    '94e9b0f60c4c7d486222b61e8f595c7f14e2dbd877ba14c1ca351ee89cd46e55'
$powmod = Get-CheckedFile 'powmod.m' `
    'https://people.cs.kuleuven.be/~dirk.nuyens/fast-cbc/powmod.m' `
    '0311c957c8d71df338438f3cc9a642a4c8e4d02f95824ba3cd465ee4466bb16c'
$wheel = Get-CheckedFile 'python-flint-0.8.0-win64.whl' `
    'https://files.pythonhosted.org/packages/c8/72/1efb1b88319269e8cb6bba709cf95b9de54b6dc0129ad623b6dd6e630433/python_flint-0.8.0-cp311-cp311-win_amd64.whl' `
    '4a52f084051d5e1fd1279ec11435c9d389eaae5f9937015818cea4a97d715d17'
$mingw = Get-CheckedFile 'mingw81.zip' `
    'https://ssd.mathworks.com/supportfiles/downloads/SupportPackageThirdParty/R2023b/mingw_w64/win64/mingw81.zip' `
    '382a1f8499aba2b21940b67d88fc4413b0b93a3c21c207178f6102b3cadc3a2d'

$functions = Join-Path $PSScriptRoot 'functions'
$grossMembers = @(
    'SupplementaryMaterial/deterministic_mr1l_approx_get_lattice_sizes.m'
    'SupplementaryMaterial/get_reconstructable_freq.m'
)
Expand-Members $gross (Join-Path $functions 'MultipleRank1Lattice') $grossMembers
Copy-Item -LiteralPath $generator,$powmod -Destination $functions -Force

$wheelMembers = @(
    'python_flint.libs/libflint.21.0.0-67b518837b39088fce997dec2b675a6a.dll'
    'python_flint.libs/libgcc_s_seh-1-f2b6825d483bdf14050493af93b5997d.dll'
    'python_flint.libs/libgmp-10-a8a2bf81a7885e882dc5b408a196b72f.dll'
    'python_flint.libs/libmpfr-6-58075fd5a9688dfd00e645316ed6a020.dll'
    'python_flint.libs/libwinpthread-1-e01b8e85fd67c2b861f64d4ccc7df607.dll'
)
$mingwMembers = @('bin/libgcc_s_seh-1.dll','bin/libgomp-1.dll','bin/libwinpthread-1.dll')
$staging = Join-Path $ArchiveDirectory 'extracted'
Expand-Members $wheel $staging $wheelMembers
Expand-Members $mingw $staging $mingwMembers
$runtime = Join-Path $functions 'cbc_precision/native/win64/runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
foreach ($name in ($wheelMembers + $mingwMembers)) {
    Copy-Item -LiteralPath (Join-Path $staging $name) -Destination $runtime -Force
}
Write-Output 'The upstream MATLAB functions and Windows CBC DLLs are ready.'
