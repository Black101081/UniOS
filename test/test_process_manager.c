#include "../kernel/process_manager.h"

void test_process_manager() {
    printf("Bắt đầu kiểm thử Process Manager...\n");

    create_process("Process 1");
    create_process("Process 2");

    terminate_process(1);

    list_processes();

    printf("Kiểm thử Process Manager hoàn thành.\n");
}

int main() {
    test_process_manager();
    return 0;
}