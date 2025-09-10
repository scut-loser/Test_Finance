package com.financial.service;

import com.financial.entity.PredictionResult;
import com.financial.repository.PredictionResultRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class PredictionResultService {
    private final PredictionResultRepository repo;
    public PredictionResultService(PredictionResultRepository repo){ this.repo=repo; }

    public PredictionResult save(PredictionResult pr){ return repo.save(pr); }
    public List<PredictionResult> latestBySymbol(String symbol){
        return repo.findTop50BySymbolOrderByPredictionTimeDesc(symbol);
    }
}