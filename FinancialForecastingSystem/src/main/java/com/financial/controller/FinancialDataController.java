package com.financial.controller;

import com.financial.entity.FinancialData;
import com.financial.service.FinancialDataService;
import com.financial.util.ImportExportUtil;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

@RestController
@RequestMapping("/financial-data")
@CrossOrigin(origins="*")
public class FinancialDataController {
    private final FinancialDataService service;
    public FinancialDataController(FinancialDataService service){ this.service=service; }

    @GetMapping
    public ResponseEntity<Page<FinancialData>> page(
            @RequestParam(defaultValue="0") int page,
            @RequestParam(defaultValue="20") int size,
            @RequestParam(required=false) String symbol){
        return ResponseEntity.ok(service.page(symbol, PageRequest.of(page, size)));
    }

    /**
     * 获取金融数据列表（分页）
     */
    @GetMapping("/time-range")
    public ResponseEntity<List<FinancialData>> range(
        @RequestParam String symbol,
        @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
        @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime){
        return ResponseEntity.ok(service.range(symbol, startTime, endTime));
    }

    /**
     * 根据时间范围查询数据
     */
    @PostMapping
    public ResponseEntity<?> create(@RequestBody FinancialData body){
        return ResponseEntity.ok(Map.of("success", true, "data", service.save(body)));
    }

    /**
     * 更新金融数据
     */
    @PutMapping("/{id}")
    public ResponseEntity<?> update(@PathVariable Long id, @RequestBody FinancialData body){
        body.setId(id);
        return ResponseEntity.ok(Map.of("success", true, "data", service.save(body)));
    }

    /**
     * 获取异常数据
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<?> delete(@PathVariable Long id){
        service.delete(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    /** 获取已有的合约(symbol)列表 */
    @GetMapping("/symbols")
    public ResponseEntity<?> symbols(){
        return ResponseEntity.ok(Map.of("success", true, "data", service.symbols()));
    }

    // Import CSV/Excel historical data
    @PostMapping(value = "/import", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<?> importData(@RequestParam("file") MultipartFile file) {
        if (file==null || file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("success", false, "message", "文件为空"));
        }
        String filename = file.getOriginalFilename()!=null?file.getOriginalFilename().toLowerCase():"";
        try {
            ImportExportUtil.ImportResult res;
            if (filename.endsWith(".csv")) {
                res = ImportExportUtil.parseCsv(file.getInputStream());
            } else if (filename.endsWith(".xlsx") || filename.endsWith(".xls")) {
                res = ImportExportUtil.parseExcel(file.getInputStream());
            } else {
                return ResponseEntity.badRequest().body(Map.of("success", false, "message", "仅支持CSV或Excel"));
            }
            Map<String, Object> stats = service.bulkImport(res.records);
            Map<String, Object> resp = new HashMap<>();
            resp.put("success", true);
            resp.put("stats", stats);
            resp.put("errorRows", res.errors);
            return ResponseEntity.ok(resp);
        } catch (Exception ex) {
            return ResponseEntity.internalServerError().body(Map.of("success", false, "message", ex.getMessage()));
        }
    }

    // Export by symbol and time range with optional format
    @GetMapping(value = "/export")
    public ResponseEntity<byte[]> exportData(
            @RequestParam String symbol,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime,
            @RequestParam(defaultValue = "csv") String format
    ) {
        try {
            List<FinancialData> list = service.range(symbol, startTime, endTime);
            String fname = "financial_data_" + symbol + "." + ("excel".equalsIgnoreCase(format) ? "xlsx" : "csv");
            byte[] content;
            MediaType mediaType;
            if ("excel".equalsIgnoreCase(format) || "xlsx".equalsIgnoreCase(format)) {
                content = ImportExportUtil.writeExcel(list);
                mediaType = MediaType.parseMediaType("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
            } else {
                content = ImportExportUtil.writeCsv(list);
                mediaType = MediaType.TEXT_PLAIN;
            }
            return ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=" + fname)
                    .contentType(mediaType)
                    .body(content);
        } catch (Exception ex) {
            return ResponseEntity.internalServerError().build();
        }
    }
}