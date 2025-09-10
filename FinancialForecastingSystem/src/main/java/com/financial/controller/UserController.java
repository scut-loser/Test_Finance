package com.financial.controller;

import com.financial.entity.User;
import com.financial.service.UserService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.financial.util.JwtUtil;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/users")
@CrossOrigin(origins="*")
public class UserController {
    private final UserService userService;
    private final JwtUtil jwtUtil;

    public UserController(UserService userService, JwtUtil jwtUtil) {
        this.userService = userService;
        this.jwtUtil = jwtUtil;
    }
    /**
     * 登录
     */
    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody Map<String,String> req){
        try {
            User u = userService.login(req.get("username"), req.get("password"));
            
            // 生成JWT token
            String token = jwtUtil.generateToken(u.getUsername(), u.getId());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "登录成功");
            response.put("user", u);
            response.put("token", token);
            response.put("tokenType", "Bearer");
            response.put("expiresIn", 24 * 60 * 60); // 24小时，单位：秒
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                "success", false,
                "message", e.getMessage()
            ));
        }
    }

    /**
     * 注册
     */
    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody Map<String,String> req){
        User u = userService.register(req.get("username"), req.get("password"));
        return ResponseEntity.ok(Map.of("success", true, "message", "注册成功", "user", u));
    }

}