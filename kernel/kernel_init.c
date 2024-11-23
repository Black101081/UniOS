#include "memory_manager.c"
#include "process_manager.c"
#include "file_manager.c"
#include "sandbox_manager.c"

void kernel_main() {
    printf("UniOS Kernel Initialized!");

    // Test quản lý bộ nhớ
    void *mem = allocate_memory(1024);

    // Test quản lý tiến trình
    create_process("Process1");
    create_sandbox("Process1"); // Tạo sandbox cho tiến trình

    terminate_sandbox(1); // Kết thúc sandbox
    terminate_process(1); // Kết thúc tiến trình

    free_memory(mem);

    while (1) {
        // Kernel loop
    }
}
