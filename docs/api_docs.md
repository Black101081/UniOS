# UniOS API Documentation

## Chức năng API

### 1. Quản lý sandbox
- **api_create_sandbox(const char *process_name):**
  - Tạo sandbox cho tiến trình chỉ định.
- **api_terminate_sandbox(int sandbox_id):**
  - Kết thúc sandbox với ID được chỉ định.

Cách sử dụng:
```c
api_create_sandbox("Process1");
api_terminate_sandbox(1);
```
