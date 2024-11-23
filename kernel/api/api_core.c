#include <stdio.h>

// Giao tiếp API và Kernel
void api_request_memory(size_t size) {
    printf("API: Yêu cầu cấp phát bộ nhớ %zu bytes.\n", size);
    // Gửi yêu cầu tới kernel (giả lập)
}

void api_create_process(const char *name) {
    printf("API: Yêu cầu tạo tiến trình %s.\n", name);
}
