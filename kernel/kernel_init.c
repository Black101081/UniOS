#include "memory_manager.c"
#include "process_manager.c"
#include "file_manager.c"

void kernel_main() {
    printf("UniOS Kernel Initialized!");

    // Test quản lý bộ nhớ
    void *mem = allocate_memory(1024);

    // Test quản lý tiến trình
    create_process("TestProcess1");
    create_process("TestProcess2");
    terminate_process(1);

    // Test quản lý tập tin
    create_file("example.txt");
    delete_file("example.txt");

    free_memory(mem);

    while (1) {
        // Kernel loop
    }
}
