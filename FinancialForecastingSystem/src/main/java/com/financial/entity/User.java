package com.financial.entity;

import javax.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "users")
public class User {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable=false, unique=true, length=64)
    private String username;

    @Column(nullable=false, length=128)
    private String password;

    @Column(name="created_time", nullable=false)
    private LocalDateTime createdTime;

    @PrePersist
    public void prePersist() {
        if (createdTime == null) createdTime = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id=id; }
    public String getUsername() { return username; }
    public void setUsername(String username) { this.username=username; }
    public String getPassword() { return password; }
    public void setPassword(String password) { this.password=password; }
    public LocalDateTime getCreatedTime() { return createdTime; }
    public void setCreatedTime(LocalDateTime createdTime) { this.createdTime=createdTime; }
}