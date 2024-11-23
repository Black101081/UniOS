#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Quản lý bộ nhớ chi tiết
void *allocate_memory(size_t size) {
    void *block = malloc(size);
    if (block) {
        printf("Allocated %zu bytes at %p
", size, block);
        memset(block, 0, size); // Khởi tạo bộ nhớ với 0
    } else {
        printf("Memory allocation failed!
");
    }
    return block;
}

void free_memory(void *block) {
    if (block) {
        free(block);
        printf("Memory freed at %p
", block);
    }
}
