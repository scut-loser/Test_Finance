package com.financial.entity;

import javax.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Entity
@Table(name = "prediction_results")
public class PredictionResult {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable=false, length=32)
    private String symbol;

    @Column(name="algorithm_name", nullable=false, length=64)
    private String algorithmName;

    @Column(name="prediction_type", length=32)
    private String predictionType;

    @Column(name="prediction_time", nullable=false)
    private LocalDateTime predictionTime;

    @Column(name="predicted_value", precision=18, scale=6)
    private BigDecimal predictedValue;

    @Column(name="confidence_score", precision=18, scale=6)
    private BigDecimal confidenceScore;

    @Column(name="is_anomaly")
    private Boolean isAnomaly;

    @Column(name="created_time", nullable=false)
    private LocalDateTime createdTime;

    @PrePersist
    public void prePersist() {
        if (createdTime == null) createdTime = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id=id; }
    public String getSymbol() { return symbol; }
    public void setSymbol(String symbol) { this.symbol=symbol; }
    public String getAlgorithmName() { return algorithmName; }
    public void setAlgorithmName(String algorithmName) { this.algorithmName=algorithmName; }
    public String getPredictionType() { return predictionType; }
    public void setPredictionType(String predictionType) { this.predictionType=predictionType; }
    public LocalDateTime getPredictionTime() { return predictionTime; }
    public void setPredictionTime(LocalDateTime predictionTime) { this.predictionTime=predictionTime; }
    public BigDecimal getPredictedValue() { return predictedValue; }
    public void setPredictedValue(BigDecimal predictedValue) { this.predictedValue=predictedValue; }
    public BigDecimal getConfidenceScore() { return confidenceScore; }
    public void setConfidenceScore(BigDecimal confidenceScore) { this.confidenceScore=confidenceScore; }
    public Boolean getIsAnomaly() { return isAnomaly; }
    public void setIsAnomaly(Boolean isAnomaly) { this.isAnomaly=isAnomaly; }
    public LocalDateTime getCreatedTime() { return createdTime; }
    public void setCreatedTime(LocalDateTime createdTime) { this.createdTime=createdTime; }
}