package com.financial.util;

import com.financial.entity.FinancialData;
import com.financial.entity.PredictionResult;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * 数据处理工具类
 */
public class DataUtil {
    
    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    
    /**
     * 格式化时间
     */
    public static String formatDateTime(LocalDateTime dateTime) {
        if (dateTime == null) {
            return "";
        }
        return dateTime.format(DATE_FORMATTER);
    }
    
    /**
     * 解析时间字符串
     */
    public static LocalDateTime parseDateTime(String dateTimeStr) {
        if (dateTimeStr == null || dateTimeStr.trim().isEmpty()) {
            return null;
        }
        try {
            return LocalDateTime.parse(dateTimeStr, DATE_FORMATTER);
        } catch (Exception e) {
            return null;
        }
    }
    
    /**
     * 格式化价格
     */
    public static String formatPrice(BigDecimal price) {
        if (price == null) {
            return "¥0.00";
        }
        return String.format("¥%.2f", price);
    }
    
    /**
     * 格式化百分比
     */
    public static String formatPercentage(BigDecimal value) {
        if (value == null) {
            return "0.00%";
        }
        return String.format("%.2f%%", value.multiply(new BigDecimal("100")));
    }
    
    /**
     * 计算移动平均线
     */
    public static BigDecimal calculateMA(List<FinancialData> dataList, int period) {
        if (dataList == null || dataList.size() < period) {
            return BigDecimal.ZERO;
        }
        
        BigDecimal sum = BigDecimal.ZERO;
        for (int i = 0; i < period; i++) {
            FinancialData data = dataList.get(i);
            if (data.getBidPrice() != null) {
                sum = sum.add(data.getBidPrice());
            }
        }
        
        return sum.divide(new BigDecimal(period), 4, BigDecimal.ROUND_HALF_UP);
    }
    
    /**
     * 计算涨跌幅
     */
    public static BigDecimal calculateChangeRate(BigDecimal currentPrice, BigDecimal previousPrice) {
        if (currentPrice == null || previousPrice == null || previousPrice.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        
        return currentPrice.subtract(previousPrice)
                .divide(previousPrice, 4, BigDecimal.ROUND_HALF_UP)
                .multiply(new BigDecimal("100"));
    }
    
    /**
     * 计算成交量变化率
     */
    public static BigDecimal calculateVolumeChangeRate(Long currentVolume, Long previousVolume) {
        if (currentVolume == null || previousVolume == null || previousVolume == 0) {
            return BigDecimal.ZERO;
        }
        
        return new BigDecimal(currentVolume - previousVolume)
                .divide(new BigDecimal(previousVolume), 4, BigDecimal.ROUND_HALF_UP)
                .multiply(new BigDecimal("100"));
    }
    
    /**
     * 获取数据统计信息
     */
    public static Map<String, Object> getDataStatistics(List<FinancialData> dataList) {
        Map<String, Object> statistics = new HashMap<>();
        
        if (dataList == null || dataList.isEmpty()) {
            statistics.put("count", 0);
            statistics.put("minPrice", BigDecimal.ZERO);
            statistics.put("maxPrice", BigDecimal.ZERO);
            statistics.put("avgPrice", BigDecimal.ZERO);
            statistics.put("totalVolume", 0L);
            return statistics;
        }
        
        int count = dataList.size();
        BigDecimal minPrice = null;
        BigDecimal maxPrice = null;
        BigDecimal sumPrice = BigDecimal.ZERO;
        long totalVolume = 0L;
        
        for (FinancialData data : dataList) {
            if (data.getBidPrice() != null) {
                if (minPrice == null || data.getBidPrice().compareTo(minPrice) < 0) {
                    minPrice = data.getBidPrice();
                }
                if (maxPrice == null || data.getBidPrice().compareTo(maxPrice) > 0) {
                    maxPrice = data.getBidPrice();
                }
                sumPrice = sumPrice.add(data.getBidPrice());
            }

            long volume = (data.getBidExecutedQty() != null ? data.getBidExecutedQty() : 0) + 
                        (data.getAskExecutedQty() != null ? data.getAskExecutedQty() : 0);
            if (volume > 0) {
                totalVolume += volume;
            }
        }
        
        BigDecimal avgPrice = count > 0 ? sumPrice.divide(new BigDecimal(count), 4, BigDecimal.ROUND_HALF_UP) : BigDecimal.ZERO;
        
        statistics.put("count", count);
        statistics.put("minPrice", minPrice != null ? minPrice : BigDecimal.ZERO);
        statistics.put("maxPrice", maxPrice != null ? maxPrice : BigDecimal.ZERO);
        statistics.put("avgPrice", avgPrice);
        statistics.put("totalVolume", totalVolume);
        
        return statistics;
    }
    
    /**
     * 获取预测结果统计信息
     */
    public static Map<String, Object> getPredictionStatistics(List<PredictionResult> predictionList) {
        Map<String, Object> statistics = new HashMap<>();
        
        if (predictionList == null || predictionList.isEmpty()) {
            statistics.put("count", 0);
            statistics.put("avgConfidence", BigDecimal.ZERO);
            statistics.put("anomalyCount", 0);
            return statistics;
        }
        
        int count = predictionList.size();
        BigDecimal sumConfidence = BigDecimal.ZERO;
        int anomalyCount = 0;
        
        for (PredictionResult prediction : predictionList) {
            if (prediction.getConfidenceScore() != null) {
                sumConfidence = sumConfidence.add(prediction.getConfidenceScore());
            }
            
            if (prediction.getIsAnomaly() != null && prediction.getIsAnomaly()) {
                anomalyCount++;
            }
        }
        
        BigDecimal avgConfidence = count > 0 ? sumConfidence.divide(new BigDecimal(count), 4, BigDecimal.ROUND_HALF_UP) : BigDecimal.ZERO;
        
        statistics.put("count", count);
        statistics.put("avgConfidence", avgConfidence);
        statistics.put("anomalyCount", anomalyCount);
        
        return statistics;
    }
    
    /**
     * 验证期货代码格式
     */
    public static boolean isValidSymbol(String symbol) {
        if (symbol == null || symbol.trim().isEmpty()) {
            return false;
        }
        
        // 简单的期货代码格式验证
        String trimmedSymbol = symbol.trim();
        return trimmedSymbol.length() >= 2 && trimmedSymbol.length() <= 20;
    }
    
    /**
     * 验证价格数据
     */
    public static boolean isValidPrice(BigDecimal price) {
        return price != null && price.compareTo(BigDecimal.ZERO) >= 0;
    }
    
    /**
     * 验证成交量数据
     */
    public static boolean isValidVolume(Long volume) {
        return volume != null && volume >= 0;
    }
}
