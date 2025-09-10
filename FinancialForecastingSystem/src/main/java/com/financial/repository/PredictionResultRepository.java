package com.financial.repository;

import com.financial.entity.PredictionResult;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.time.LocalDateTime;

public interface PredictionResultRepository extends JpaRepository<PredictionResult, Long> {
    List<PredictionResult> findTop50BySymbolOrderByPredictionTimeDesc(String symbol);
    List<PredictionResult> findBySymbolAndPredictionTimeBetweenOrderByPredictionTimeAsc(String symbol, LocalDateTime start, LocalDateTime end);
    List<PredictionResult> findBySymbolAndIsAnomalyTrueAndPredictionTimeBetweenOrderByPredictionTimeAsc(String symbol, LocalDateTime start, LocalDateTime end);
}