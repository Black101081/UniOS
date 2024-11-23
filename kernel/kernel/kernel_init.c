#include "memory_manager.c"
#include "process_manager.c"

void kernel_main() {
    printf("UniOS Kernel Initialized!\n");

    // Ví dụ sử dụng
    void *mem = allocate_memory(1024);
    create_process("TestProcess");

    free_memory(mem);
    terminate_process("TestProcess");

    while (1) {
        // Kernel loop
    }
}
