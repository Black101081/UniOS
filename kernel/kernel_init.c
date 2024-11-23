#include "memory_manager.c"
#include "process_manager.c"

void kernel_main() {
    printf("UniOS Kernel Initialized!
");

    // Test quản lý bộ nhớ
    void *mem = allocate_memory(1024);

    // Test quản lý tiến trình
    create_process("TestProcess1");
    create_process("TestProcess2");
    terminate_process(1);

    free_memory(mem);

    while (1) {
        // Kernel loop
    }
}
