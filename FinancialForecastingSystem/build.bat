@echo off
echo ========================================
echo 金融时序数据预测系统构建脚本
echo ========================================

echo.
echo 1. 清理项目...
call mvn clean

echo.
echo 2. 编译项目...
call mvn compile

echo.
echo 3. 运行测试...
call mvn test

echo.
echo 4. 打包项目...
call mvn package -DskipTests

echo.
echo 5. 构建完成！
echo 可执行文件位置: target\financial-forecasting-system-1.0.0.jar
echo.
echo 运行命令: java -jar target\financial-forecasting-system-1.0.0.jar
echo.

pause
