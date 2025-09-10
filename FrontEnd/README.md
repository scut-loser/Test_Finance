# FrontEnd 使用说明

## 1. 依赖安装（Windows）

方式 A：PowerShell（推荐）

```powershell
cd FrontEnd
Set-ExecutionPolicy -Scope Process RemoteSigned
./SETUP_WIN.ps1
```

方式 B：批处理脚本（双击也可）

```bat
cd FrontEnd
SETUP_WIN.bat
```

以上脚本会：
- 创建并激活虚拟环境 `.venv`
- 安装 `requirements.txt`
- 生成 `.env`（如不存在），默认 `BACKEND_BASE_URL=http://localhost:8080`

## 2. 手工安装（可选）
```bash
cd FrontEnd
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env  # 若需要
```

## 3. 运行
- 先确保 Java 后端已启动在 `http://localhost:8080`
- 启动示例脚本：
```bash
# Windows
.\.venv\Scripts\activate
python FrontEnd\main.py
```

## 4. 配置
- `.env` 文件：
```
BACKEND_BASE_URL=http://localhost:8080
# BACKEND_TOKEN=your_jwt_token_here
```

## 5. 常见问题
- 若 PyQt6 安装较慢，可换源（pip 国内镜像）或先升级 pip。
- 若端口/地址不同，改 `.env` 中的 `BACKEND_BASE_URL`。
