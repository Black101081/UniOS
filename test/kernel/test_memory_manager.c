#include <assert.h>
#include <string.h>
#include "kernel/core/memory/memory_manager.h"

#define TEST_MEMORY_SIZE (10 * 1024) // 10KB for testing

void test_basic_allocation() {
    static uint8_t test_memory[1024];
    init_memory_manager(test_memory, sizeof(test_memory));
    
    // Test nhỏ trước
    void* ptr1 = memory_alloc(10);
    assert(ptr1 != NULL);
    memset(ptr1, 0x55, 10);
    
    void* ptr2 = memory_alloc(20);
    assert(ptr2 != NULL);
    assert(ptr2 > ptr1);
    
    memory_free(ptr1);
    memory_free(ptr2);
    
    printf("Basic allocation test passed!\n");
}

void test_fragmentation_and_defragmentation() {
    static uint8_t test_memory[TEST_MEMORY_SIZE];
    init_memory_manager(test_memory, TEST_MEMORY_SIZE);
    
    void* ptrs[5];
    // Cấp phát 5 blocks
    for (int i = 0; i < 5; i++) {
        ptrs[i] = memory_alloc(100);
        assert(ptrs[i] != NULL);
    }
    
    // Giải phóng 3 block liền kề
    for (int i = 0; i < 3; i++) {
        memory_free(ptrs[i]);
    }
    
    MemoryStats stats_before = get_memory_stats();
    defragment_memory();
    MemoryStats stats_after = get_memory_stats();
    
    assert(stats_after.free_blocks < stats_before.free_blocks);
    assert(check_memory_integrity());
    
    printf("Fragmentation tests passed!\n");
}

void test_memory_corruption_detection() {
    static uint8_t test_memory[TEST_MEMORY_SIZE];
    init_memory_manager(test_memory, TEST_MEMORY_SIZE);
    
    printf("Allocating memory...\n");
    void* ptr = memory_alloc(100);
    assert(ptr != NULL);
    printf("Memory allocated at %p\n", ptr);
    
    // Thử nghiệm ghi vào vùng nhớ được cấp phát trước
    uint8_t* safe_ptr = (uint8_t*)ptr;
    memset(safe_ptr, 0xAA, 100); // Ghi vào vùng nhớ an toàn
    
    // Sau đó mới thử nghiệm ghi vượt quá
    uint8_t* bad_ptr = (uint8_t*)ptr;
    bad_ptr[100] = 0xAA; // Ghi vượt quá vùng được cấp phát
    
    bool corruption_detected = !check_memory_integrity();
    assert(corruption_detected);
    
    memory_free(ptr);
    printf("Memory corruption detection tests passed!\n");
}

int main() {
    test_basic_allocation();
    test_fragmentation_and_defragmentation();
    test_memory_corruption_detection();
    
    printf("\nAll memory tests passed successfully!\n");
    return 0;
}