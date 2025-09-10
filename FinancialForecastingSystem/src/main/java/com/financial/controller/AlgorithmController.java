package com.financial.controller;

import com.financial.service.AlgorithmService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 算法服务REST API控制器
 */
@RestController
@RequestMapping("/algorithms")
@CrossOrigin(origins = "*")
public class AlgorithmController {

    @Autowired
    private AlgorithmService algorithmService;

    /**
     * 获取可用算法列表
     */
    @GetMapping("/available")
    public ResponseEntity<List<String>> getAvailableAlgorithms() {
        // List<String> algorithms = algorithmService.getAvailableAlgorithms();
        // return ResponseEntity.ok(algorithms);
        return ResponseEntity.ok(List.of("FINAL_MODEL"));
    }

    /**
     * 执行本地预测
     */
    @PostMapping("/predict/local")
    public ResponseEntity<Map<String, Object>> executeLocalPrediction(@RequestBody Map<String, Object> request) {
        try {
            String symbol = (String) request.get("symbol");
            String algorithmName = (String) request.getOrDefault("algorithmName", "FINAL_MODEL");
            String predictionType = (String) request.get("predictionType");
            String feature = (String) request.getOrDefault("feature", null);
            
            Map<String, Object> result = algorithmService.executeLocalPrediction(symbol, algorithmName, feature);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "本地预测执行成功");
            response.put("result", result);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "本地预测执行失败: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }

    /**
     * 执行云端预测(后续完善)
     */
    /**
    @PostMapping("/predict/cloud")
    public ResponseEntity<Map<String, Object>> executeCloudPrediction(@RequestBody Map<String, Object> request) {
        try {
            String symbol = (String) request.get("symbol");
            String algorithmName = (String) request.getOrDefault("algorithmName", "FINAL_MODEL");
            String predictionType = (String) request.get("predictionType");
            
            Map<String, Object> result = algorithmService.executeCloudPrediction(symbol, algorithmName, predictionType);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "云端预测执行成功");
            response.put("result", result);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "云端预测执行失败: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }
    */

    /**
     * 执行异常检测
     */
    @PostMapping("/anomaly-detection")
    public ResponseEntity<Map<String, Object>> executeAnomalyDetection(@RequestBody Map<String, Object> request) {
        try {
            String symbol = (String) request.get("symbol");
            String algorithmName = (String) request.getOrDefault("algorithmName", "FINAL_MODEL");
            
            Map<String, Object> result = algorithmService.executeAnomalyDetection(symbol, algorithmName);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "异常检测执行成功");
            response.put("result", result);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "异常检测执行失败: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }

    /**
     * 获取预测历史
     */
    @GetMapping("/predictions/history")
    public ResponseEntity<Map<String, Object>> getPredictionHistory(
            @RequestParam(required = false) String symbol,
            @RequestParam(required = false) String algorithmName,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        
        try {
            // 这里需要调用PredictionResultService来获取历史数据
            // 暂时返回模拟数据
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "获取预测历史成功");
            response.put("data", List.of());
            response.put("total", 0);
            response.put("page", page);
            response.put("size", size);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "获取预测历史失败: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }
}
