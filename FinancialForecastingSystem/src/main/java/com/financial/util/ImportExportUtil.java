package com.financial.util;

import com.financial.entity.FinancialData;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.*;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.temporal.ChronoField;
import java.util.ArrayList;
import java.util.List;

public class ImportExportUtil {
    private static final DateTimeFormatter ISO = DateTimeFormatter.ISO_DATE_TIME;

    public static class ImportResult {
        public final List<FinancialData> records;
        public final List<String> errors;
        public ImportResult(List<FinancialData> records, List<String> errors){
            this.records = records; this.errors = errors;
        }
    }

    public static ImportResult parseCsv(InputStream in) throws IOException {
        List<FinancialData> list = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        Iterable<CSVRecord> records = CSVFormat.DEFAULT
                .withFirstRecordAsHeader()
                .withIgnoreHeaderCase()
                .withTrim()
                .parse(new InputStreamReader(in, StandardCharsets.UTF_8));
        int row = 1;
        for (CSVRecord r : records) {
            row++;
            try {
                FinancialData fd = fromRow(
                        safeGet(r, "symbol", "\ufeffsymbol"),
                        // accept both date_time and datetime
                        safeGet(r, "date_time", "datetime"),
                        safeGet(r, "bid_price"),
                        safeGet(r, "bid_order_qty"),
                        safeGet(r, "bid_executed_qty"),
                        safeGet(r, "ask_order_qty"),
                        safeGet(r, "ask_executed_qty")
                );
                list.add(fd);
            } catch (Exception ex) {
                errors.add("CSV row " + row + ": " + ex.getMessage());
            }
        }
        return new ImportResult(list, errors);
    }

    public static ImportResult parseExcel(InputStream in) throws IOException {
        List<FinancialData> list = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        try(Workbook wb = new XSSFWorkbook(in)){
            Sheet sheet = wb.getSheetAt(0);
            int rows = sheet.getPhysicalNumberOfRows();
            // Assume header at row 0
            for(int i=1;i<rows;i++){
                Row r = sheet.getRow(i);
                if(r==null) continue;
                try{
                    FinancialData fd = fromRow(
                            getString(r,0),
                            getString(r,1),
                            getString(r,2),
                            getString(r,3),
                            getString(r,4),
                            getString(r,5),
                            getString(r,6)
                    );
                    list.add(fd);
                }catch(Exception ex){
                    errors.add("Excel row " + (i+1) + ": " + ex.getMessage());
                }
            }
        }
        return new ImportResult(list, errors);
    }

    private static String getString(Row row, int idx){
        Cell cell = row.getCell(idx);
        if(cell==null) return null;
        cell.setCellType(CellType.STRING);
        return cell.getStringCellValue();
    }

    private static FinancialData fromRow(String symbol, String dateTime, String bidPrice,
                                         String bidOrderQty, String bidExecutedQty,
                                         String askOrderQty, String askExecutedQty){
        if(symbol==null || symbol.isBlank()) throw new IllegalArgumentException("symbol is blank");
        if(dateTime==null || dateTime.isBlank()) throw new IllegalArgumentException("date_time is blank");
        FinancialData fd = new FinancialData();
        fd.setSymbol(symbol.trim());
        fd.setDateTime(parseDateTimeFlexible(dateTime.trim()));
        if(bidPrice!=null && !bidPrice.isBlank()) fd.setBidPrice(new BigDecimal(bidPrice.trim()));
        if(bidOrderQty!=null && !bidOrderQty.isBlank()) fd.setBidOrderQty(Long.parseLong(bidOrderQty.trim()));
        if(bidExecutedQty!=null && !bidExecutedQty.isBlank()) fd.setBidExecutedQty(Long.parseLong(bidExecutedQty.trim()));
        if(askOrderQty!=null && !askOrderQty.isBlank()) fd.setAskOrderQty(Long.parseLong(askOrderQty.trim()));
        if(askExecutedQty!=null && !askExecutedQty.isBlank()) fd.setAskExecutedQty(Long.parseLong(askExecutedQty.trim()));
        return fd;
    }

    private static String safeGet(CSVRecord r, String... names){
        for(String n : names){
            try { return r.get(n); } catch (IllegalArgumentException ignore) {}
        }
        return null;
    }

    private static LocalDateTime parseDateTimeFlexible(String s){
        // Try ISO first
        try { return LocalDateTime.parse(s, ISO); } catch (Exception ignore) {}
        // Try with space between date and time and optional fraction and timezone offset
        DateTimeFormatter fmt = new DateTimeFormatterBuilder()
                .appendPattern("yyyy-MM-dd HH:mm:ss")
                .optionalStart()
                .appendFraction(ChronoField.NANO_OF_SECOND, 1, 9, true)
                .optionalEnd()
                .optionalStart()
                .appendOffset("+HH:MM", "+00:00")
                .optionalEnd()
                .toFormatter();
        try {
            // If string has offset, parse as OffsetDateTime then convert to LocalDateTime
            if (s.contains("+") || s.contains("-")) {
                return OffsetDateTime.parse(s, fmt).toLocalDateTime();
            }
            return LocalDateTime.parse(s, fmt);
        } catch (Exception ex) {
            throw new IllegalArgumentException("invalid date_time format: " + s);
        }
    }

    public static byte[] writeCsv(List<FinancialData> list) throws IOException {
        String[] header = new String[]{
                "symbol","date_time","bid_price","bid_order_qty","bid_executed_qty","ask_order_qty","ask_executed_qty"
        };
        try(ByteArrayOutputStream bout = new ByteArrayOutputStream();
            OutputStreamWriter writer = new OutputStreamWriter(bout);
            CSVPrinter printer = new CSVPrinter(writer, CSVFormat.DEFAULT.withHeader(header))){
            for(FinancialData fd : list){
                printer.printRecord(
                        fd.getSymbol(),
                        fd.getDateTime()!=null?fd.getDateTime().format(ISO):null,
                        fd.getBidPrice(),
                        fd.getBidOrderQty(),
                        fd.getBidExecutedQty(),
                        fd.getAskOrderQty(),
                        fd.getAskExecutedQty()
                );
            }
            printer.flush();
            return bout.toByteArray();
        }
    }

    public static byte[] writeExcel(List<FinancialData> list) throws IOException {
        try(Workbook wb = new XSSFWorkbook(); ByteArrayOutputStream bout = new ByteArrayOutputStream()){
            Sheet sheet = wb.createSheet("data");
            String[] header = new String[]{
                    "symbol","date_time","bid_price","bid_order_qty","bid_executed_qty","ask_order_qty","ask_executed_qty"
            };
            Row h = sheet.createRow(0);
            for(int i=0;i<header.length;i++){
                h.createCell(i).setCellValue(header[i]);
            }
            int r = 1;
            for(FinancialData fd : list){
                Row row = sheet.createRow(r++);
                row.createCell(0).setCellValue(fd.getSymbol());
                row.createCell(1).setCellValue(fd.getDateTime()!=null?fd.getDateTime().format(ISO):null);
                if(fd.getBidPrice()!=null) row.createCell(2).setCellValue(fd.getBidPrice().doubleValue());
                if(fd.getBidOrderQty()!=null) row.createCell(3).setCellValue(fd.getBidOrderQty());
                if(fd.getBidExecutedQty()!=null) row.createCell(4).setCellValue(fd.getBidExecutedQty());
                if(fd.getAskOrderQty()!=null) row.createCell(5).setCellValue(fd.getAskOrderQty());
                if(fd.getAskExecutedQty()!=null) row.createCell(6).setCellValue(fd.getAskExecutedQty());
            }
            wb.write(bout);
            return bout.toByteArray();
        }
    }
}


