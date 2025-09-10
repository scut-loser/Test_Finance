package com.financial.config;

import com.financial.util.JwtUtil;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@Component
public class AuthInterceptor implements HandlerInterceptor {
    
    private final JwtUtil jwtUtil;
    
    public AuthInterceptor(JwtUtil jwtUtil) {
        this.jwtUtil = jwtUtil;
    }
    
    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) {
        // 不需要认证的路径
        String[] publicPaths = {
            "/api/users/login",
            "/api/users/register",
            "/error"
        };
        
        String requestPath = request.getRequestURI();
        for (String publicPath : publicPaths) {
            if (requestPath.startsWith(publicPath)) {
                return true;
            }
        }
        
        // 获取Authorization header
        String authHeader = request.getHeader("Authorization");
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            return false;
        }
        
        // 验证token
        String token = authHeader.substring(7);
        if (!jwtUtil.validateToken(token)) {
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            return false;
        }
        
        return true;
    }
}