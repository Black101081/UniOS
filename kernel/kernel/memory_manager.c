#include <stdio.h>
#include <stdlib.h>

// Quản lý bộ nhớ cơ bản
void *allocate_memory(size_t size) {
    void *block = malloc(size);
    if (block) {
        printf("Allocated %zu bytes at %p\n", size, block);
    } else {
        printf("Memory allocation failed!\n");
    }
    return block;
}

void free_memory(void *block) {
    if (block) {
        free(block);
        printf("Memory freed at %p\n", block);
    }
}
