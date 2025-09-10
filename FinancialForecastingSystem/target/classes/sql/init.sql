CREATE DATABASE IF NOT EXISTS financial_forecasting DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE financial_forecasting;

CREATE TABLE IF NOT EXISTS users (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(64) NOT NULL UNIQUE,
  password VARCHAR(128) NOT NULL,
  created_time DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS financial_data (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  symbol VARCHAR(32) NOT NULL,
  date_time DATETIME NOT NULL,
  bid_price DECIMAL(18,6),
  bid_order_qty BIGINT,
  bid_executed_qty BIGINT,
  ask_order_qty BIGINT,
  ask_executed_qty BIGINT,
  INDEX idx_fd_symbol_time (symbol, date_time)
);

CREATE TABLE IF NOT EXISTS prediction_results (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  symbol VARCHAR(32) NOT NULL,
  algorithm_name VARCHAR(64) NOT NULL,
  prediction_type VARCHAR(32),
  prediction_time DATETIME NOT NULL,
  predicted_value DECIMAL(18,6),
  confidence_score DECIMAL(18,6),
  is_anomaly TINYINT(1),
  created_time DATETIME NOT NULL,
  INDEX idx_pr_symbol_time (symbol, prediction_time)
);