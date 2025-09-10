# 金融时序数据预测和异常检测系统

## 项目简介

基于 Spring Boot 的金融时序预测后端，现已集成以下四类可用算法：

- 单一特征 LSTM 模型（SINGLE_LSTM）
- 单一特征 Transformer 模型（SINGLE_TRANSFORMER）
- 串联式混合模型：LSTM → Transformer（SERIAL_LSTM_TRANSFORMER）
- 特征融合型混合模型：多维特征 + 可学习位置编码（FUSION_LSTM_TRANSFORMER）

系统通过 REST API 调用本地 Python 脚本进行训练与预测，并将预测结果持久化。

## 技术架构

### 后端技术栈
- **Spring Boot 2.7.0** - 主框架
- **Spring Web** - RESTful API支持
- **FastJSON** - JSON数据处理
- **Apache Commons Lang3** - 工具类库
- **Logback + SLF4J** - 日志管理

### 已集成算法
- SINGLE_LSTM：单变量 LSTM
- SINGLE_TRANSFORMER：单变量 Transformer（含可学习位置编码）
- SERIAL_LSTM_TRANSFORMER：LSTM 编码 + Transformer 建模的串联混合
- FUSION_LSTM_TRANSFORMER：多特征输入（价格、成交量、技术指标等） + 可学习位置编码 + LSTM/Transformer 融合

## 系统功能

### 1. 用户管理
- 用户登录/注册
- 角色权限管理
- 用户信息管理

### 2. 数据管理
- 期货数据查询和筛选
- 数据统计和分析
- 异常数据标记

### 3. 预测分析
- 四类算法预测（见“已集成算法”）
- 预测结果展示
- 预测准确率评估
- 批量预测功能

### 4. 异常检测
- 实时异常检测
- 异常数据标记
- 异常分布分析
- 检测阈值设置

### 5. 系统设置
- 算法参数设置
- 系统参数配置
- 用户偏好设置

## 项目结构

```
FinancialForecastingSystem/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/financial/
│   │   │       ├── FinancialForecastingApplication.java  # 主启动类
│   │   │       ├── config/                               # 配置类
│   │   │       ├── controller/                           # API控制器
│   │   │       ├── service/                              # 服务层
│   │   │       └── util/                                 # 工具类
│   │   └── resources/
│   │       └── application.yml                           # 配置文件
│   └── test/                                             # 测试代码
├── pom.xml                                               # Maven配置
└── README.md                                             # 项目说明
```

## 安装和运行

### 环境要求
- JDK 11+
- Maven 3.6+

### 安装步骤

1. **克隆项目**
   ```bash
   git clone [项目地址]
   cd FinancialForecastingSystem
   ```

2. **编译项目**
   ```bash
   mvn clean compile
   ```

3. **运行项目**
   ```bash
   mvn spring-boot:run
   ```

4. **打包成可执行文件**
   ```bash
   mvn clean package
   java -jar target/financial-forecasting-system-1.0.0.jar
   ```

## API 接口说明

### 用户管理接口
- `POST /api/users/login` - 用户登录
- `POST /api/users/register` - 用户注册

### 金融数据接口
- `GET /api/financial-data` - 获取金融数据列表
- `GET /api/financial-data/{id}` - 获取指定数据
- `GET /api/financial-data/latest/{symbol}` - 获取最新数据
- `GET /api/financial-data/time-range` - 按时间范围查询
- `GET /api/financial-data/statistics` - 获取统计信息
- `GET /api/financial-data/anomalies` - 获取异常数据

### 算法服务接口
- `GET /algorithms/available` 获取可用算法列表
  - 返回：[`SINGLE_LSTM`, `SINGLE_TRANSFORMER`, `SERIAL_LSTM_TRANSFORMER`, `FUSION_LSTM_TRANSFORMER`]

- `POST /algorithms/predict/local` 执行本地预测
  - 请求体示例：
    ```json
    {
      "symbol": "IF2409",
      "algorithmName": "SINGLE_LSTM",
      "feature": "bid_price"  
    }
    ```
    - 说明：`feature` 仅在单一特征或串联式模型中生效；融合模型自动使用多维特征。
  - 响应示例：
    ```json
    {
      "success": true,
      "result": {
        "predicted_value": 3821.5,
        "confidence_score": 0.93,
        "upper_bound": 3830.2,
        "lower_bound": 3812.7,
        "mse": 1.23,
        "mae": 0.84,
        "mape": 0.12,
        "r2": 0.91,
        "model_version": "algos-1.1.0",
        "algorithm": "SINGLE_LSTM",
        "feature": "bid_price",
        "prediction_id": 123,
        "saved_to_database": true
      }
    }
    ```

- `GET /algorithms/info/{algorithmName}` 获取算法说明
  - 示例：`/algorithms/info/SERIAL_LSTM_TRANSFORMER`

  - `POST /algorithms/predict/cloud`（占位）
  - `POST /algorithms/anomaly-detection`（占位/示例返回）
  - `GET /algorithms/predictions/history`（占位/示例返回）

## 算法与数据说明

- 本地脚本：`FinancialForecastingSystem/models/local_model.py`
- 主要参数：
  - `--algorithm`：四选一
  - `--feature`：单特征/串联系统使用，默认 `bid_price`
  - `--window`：滑动窗口长度（默认 30）
  - `--epochs`：训练轮次（默认 50）
  - `--data`：CSV 数据路径，需包含 `datetime` 与特征列
  - 多维特征默认包含：`bid_price`, `bid_order_qty`, `bid_executed_qty`, `ask_order_qty`, `ask_executed_qty`

- 输出：JSON，包含 `predicted_value`, `confidence_score`, 置信区间、误差指标、`model_version`, `algorithm` 等。

## 配置说明

### 算法配置（application.yml）
```yaml
financial:
  algorithm:
    local-model-path: models/local_model.py
    local-data-file: cleaned_data.csv
    python-exec: python
    prediction-window: 30
    anomaly-threshold: 0.95
    default-feature: bid_price
```

### 数据源配置
```yaml
financial:
  datasource:
    local:
      enabled: true
    cloud:
      enabled: true
      url: http://localhost:8082/api/data
      api-key: your-api-key
```

## 开发说明

### 添加新算法
1. 在 `models/local_model.py` 中实现新算法并增加 `--algorithm` 选项
2. 在 `AlgorithmService#getAvailableAlgorithms` 中登记名称
3. 在 `AlgorithmService#getAlgorithmInfo` 中补充说明
4. 如需要单特征，确保通过 `--feature` 或配置默认特征

### API接口开发
1. 创建新的Controller类
2. 定义RESTful API接口
3. 实现对应的Service方法
4. 添加接口文档注解

## 部署说明

### 开发环境
- 直接运行 `FinancialForecastingApplication.java`
- 使用IDE调试和开发

### 生产环境
- 打包成JAR文件
- 配置生产环境参数
- 使用系统服务管理

### Docker部署
```dockerfile
FROM openjdk:11-jre-slim
COPY target/financial-forecasting-system-1.0.0.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "/app.jar"]
```

## 注意事项

1. CSV 需包含 `datetime` 列，且特征列为数值类型
2. Windows 下默认 `python` 可改为 `python3` 或虚拟环境解释器
3. 生产环境请开启鉴权与限流，调整日志与连接池配置
4. 训练参数请按数据规模调整，避免占用过多资源

## 联系方式

- 开发团队：金融预测团队
- 技术支持：请联系开发团队
- 项目地址：[GitHub地址]



