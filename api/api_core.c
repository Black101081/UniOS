#include "../kernel/sandbox_manager.h"

void api_create_sandbox(const char *process_name) {
    printf("API: Tạo sandbox cho tiến trình \"%s\".\n", process_name);
    create_sandbox(process_name); // Gọi kernel
}

void api_terminate_sandbox(int sandbox_id) {
    printf("API: Kết thúc sandbox với ID %d.\n", sandbox_id);
    terminate_sandbox(sandbox_id); // Gọi kernel
}