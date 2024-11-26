#include <stdio.h>
#include "../../kernel/core/memory/memory_manager.h"

#define MEMORY_SIZE (1024 * 1024) // 1MB

int main() {
    // Cấp phát vùng nhớ để test
    static char memory_pool[MEMORY_SIZE];
    
    printf("Testing Memory Manager...\n\n");
    
    // Khởi tạo memory manager
    init_memory_manager(memory_pool, MEMORY_SIZE);
    
    // In thống kê ban đầu
    printf("Initial ");
    print_memory_stats();
    
    // Test cấp phát
    void* ptr1 = memory_alloc(1000);
    void* ptr2 = memory_alloc(2000);
    void* ptr3 = memory_alloc(3000);
    
    printf("\nAfter allocations:\n");
    print_memory_stats();
    
    // Test giải phóng
    memory_free(ptr2);
    printf("\nAfter freeing middle block:\n");
    print_memory_stats();
    
    // Test cấp phát sau khi giải phóng
    void* ptr4 = memory_alloc(1500);
    printf("\nAfter new allocation:\n");
    print_memory_stats();
    
    return 0;
}