@echo off
echo ========================================
echo 金融时序数据预测系统启动脚本
echo ========================================

echo.
echo 检查Java环境...
java -version
if %errorlevel% neq 0 (
    echo 错误: 未找到Java环境，请先安装JDK 11+
    pause
    exit /b 1
)

echo.
echo 检查Maven环境...
mvn -version
if %errorlevel% neq 0 (
    echo 错误: 未找到Maven环境，请先安装Maven 3.6+
    pause
    exit /b 1
)

echo.
echo 启动应用程序...
echo 方式1: 使用Maven运行
echo 方式2: 使用JAR文件运行
echo.
set /p choice=请选择启动方式 (1/2): 

if "%choice%"=="1" (
    echo 使用Maven启动...
    call mvn spring-boot:run
) else if "%choice%"=="2" (
    echo 检查JAR文件...
    if not exist "target\financial-forecasting-system-1.0.0.jar" (
        echo 错误: 未找到JAR文件，请先运行 build.bat 构建项目
        pause
        exit /b 1
    )
    echo 使用JAR文件启动...
    java -jar target\financial-forecasting-system-1.0.0.jar
) else (
    echo 无效选择，使用默认方式启动...
    call mvn spring-boot:run
)

pause
