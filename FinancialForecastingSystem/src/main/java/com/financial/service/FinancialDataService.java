package com.financial.service;

import com.financial.entity.FinancialData;
import com.financial.repository.FinancialDataRepository;
import org.apache.commons.lang3.StringUtils;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
public class FinancialDataService {
    private final FinancialDataRepository repo;
    public FinancialDataService(FinancialDataRepository repo){ this.repo=repo; }

    public Page<FinancialData> page(String symbol, Pageable pageable){
        if (StringUtils.isBlank(symbol)) return repo.findAll(pageable);
        return repo.findBySymbol(symbol, pageable);
    }

    public FinancialData byId(Long id){ return repo.findById(id).orElse(null); }

    public FinancialData latest(String symbol){ return repo.findTop1BySymbolOrderByDateTimeDesc(symbol); }

    public List<FinancialData> range(String symbol, LocalDateTime start, LocalDateTime end){
        return repo.findBySymbolAndDateTimeBetweenOrderByDateTimeAsc(symbol, start, end);
    }

    public FinancialData save(FinancialData fd){ return repo.save(fd); }
    public void delete(Long id){ repo.deleteById(id); }

    public Map<String, Object> bulkImport(List<FinancialData> records){
        if(records==null || records.isEmpty()){
            return Map.of("inserted", 0, "duplicates", 0, "errors", 0);
        }
        List<FinancialData> toSave = new ArrayList<>();
        int duplicates = 0;
        int errors = 0;
        for(FinancialData r : records){
            if(r==null || r.getSymbol()==null || r.getDateTime()==null){
                errors++;
                continue;
            }
            if(repo.existsBySymbolAndDateTime(r.getSymbol(), r.getDateTime())){
                duplicates++;
                continue;
            }
            toSave.add(r);
        }
        repo.saveAll(toSave);
        return Map.of(
                "inserted", toSave.size(),
                "duplicates", duplicates,
                "errors", errors
        );
    }

    public List<String> symbols(){
        return repo.findDistinctSymbols();
    }
}