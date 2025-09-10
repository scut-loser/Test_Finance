package com.financial.service;

import com.financial.entity.User;
import com.financial.repository.UserRepository;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class UserService {
    private final UserRepository userRepository;
    public UserService(UserRepository userRepository){ this.userRepository=userRepository; }

    public User register(String username, String password){
        if (userRepository.existsByUsername(username)) throw new RuntimeException("用户名已存在");
        User u=new User();
        u.setUsername(username);
        u.setPassword(password);
        return userRepository.save(u);
    }

    public User login(String username, String password){
        Optional<User> opt = userRepository.findByUsername(username);
        if (opt.isPresent() && opt.get().getPassword().equals(password)) return opt.get();
        throw new RuntimeException("用户名或密码错误");
    }

    public Optional<User> byId(Long id){ return userRepository.findById(id); }
    public User save(User u){ return userRepository.save(u); }
    public void delete(Long id){ userRepository.deleteById(id); }
}