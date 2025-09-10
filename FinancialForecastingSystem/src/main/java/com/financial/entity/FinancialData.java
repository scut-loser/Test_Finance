package com.financial.entity;

import javax.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Entity
@Table(name = "financial_data")
public class FinancialData {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable=false, length=32)
    private String symbol;

    @Column(name="date_time", nullable=false)
    private LocalDateTime dateTime;

    @Column(name="bid_price", precision=18, scale=6)
    private BigDecimal bidPrice;

    @Column(name="bid_order_qty")
    private Long bidOrderQty;

    @Column(name="bid_executed_qty")
    private Long bidExecutedQty;

    @Column(name="ask_order_qty")
    private Long askOrderQty;

    @Column(name="ask_executed_qty")
    private Long askExecutedQty;

    public Long getId() { return id; }
    public void setId(Long id) { this.id=id; }
    public String getSymbol() { return symbol; }
    public void setSymbol(String symbol) { this.symbol=symbol; }
    public LocalDateTime getDateTime() { return dateTime; }
    public void setDateTime(LocalDateTime dateTime) { this.dateTime=dateTime; }
    public BigDecimal getBidPrice() { return bidPrice; }
    public void setBidPrice(BigDecimal bidPrice) { this.bidPrice=bidPrice; }
    public Long getBidOrderQty() { return bidOrderQty; }
    public void setBidOrderQty(Long bidOrderQty) { this.bidOrderQty=bidOrderQty; }
    public Long getBidExecutedQty() { return bidExecutedQty; }
    public void setBidExecutedQty(Long bidExecutedQty) { this.bidExecutedQty=bidExecutedQty; }
    public Long getAskOrderQty() { return askOrderQty; }
    public void setAskOrderQty(Long askOrderQty) { this.askOrderQty=askOrderQty; }
    public Long getAskExecutedQty() { return askExecutedQty; }
    public void setAskExecutedQty(Long askExecutedQty) { this.askExecutedQty=askExecutedQty; }
}