Param(
    [string]$Python = "python",
    [string]$EnvFile = ".env"
)

Write-Host "[1/4] 检查 Python 版本..."
$pyver = & $Python --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "未找到 Python，请安装 Python 3.10+ 并加入 PATH。"
    exit 1
}
Write-Host "使用: $pyver"

Write-Host "[2/4] 创建虚拟环境 .venv ..."
& $Python -m venv .venv
if ($LASTEXITCODE -ne 0) { Write-Error "创建虚拟环境失败"; exit 1 }

Write-Host "[3/4] 激活虚拟环境并安装依赖..."
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Error "安装依赖失败"; deactivate; exit 1 }

Write-Host "[4/4] 生成 .env（若不存在）..."
if (-not (Test-Path $EnvFile)) {
    Set-Content -Path $EnvFile -Value "BACKEND_BASE_URL=http://localhost:8080`n# BACKEND_TOKEN=your_jwt_token_here"
}

Write-Host "完成。使用方法："
Write-Host "`t1) PowerShell 执行: .\\.venv\\Scripts\\Activate.ps1"
Write-Host "`t2) python FrontEnd\\main.py"
