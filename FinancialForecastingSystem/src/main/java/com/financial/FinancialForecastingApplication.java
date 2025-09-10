package com.financial;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 金融时序数据预测和异常检测系统主启动类
 * 纯Spring Boot后端应用
 */
@SpringBootApplication
public class FinancialForecastingApplication {

    public static void main(String[] args) {
        SpringApplication.run(FinancialForecastingApplication.class, args);
    }
}
