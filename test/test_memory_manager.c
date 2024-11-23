#include "../kernel/memory_manager.c"

void test_memory_manager() {
    printf("Bắt đầu kiểm thử Memory Manager...\n");
    void *block1 = allocate_memory(256);
    void *block2 = allocate_memory(128);

    free_memory(block1);
    free_memory(block2);

    printf("Kiểm thử Memory Manager hoàn thành.\n");
}

int main() {
    test_memory_manager();
    return 0;
}