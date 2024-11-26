#include "memory_manager.h"
#include <stdio.h>

static MemoryBlock* memory_list = NULL;
static void* memory_start = NULL;
static size_t total_memory_size = 0;

void init_memory_manager(void* start, size_t size) {
    memory_start = start;
    total_memory_size = size;
    
    // Khởi tạo block đầu tiên
    memory_list = (MemoryBlock*)start;
    memory_list->address = (uint32_t)start + sizeof(MemoryBlock);
    memory_list->size = size - sizeof(MemoryBlock);
    memory_list->is_free = true;
    memory_list->next = NULL;
}

void* memory_alloc(size_t size) {
    MemoryBlock* current = memory_list;
    
    // Tìm block trống đủ lớn
    while (current != NULL) {
        if (current->is_free && current->size >= size) {
            // Nếu block quá lớn, chia nó
            if (current->size > size + sizeof(MemoryBlock) + 32) {
                MemoryBlock* new_block = (MemoryBlock*)((char*)current + sizeof(MemoryBlock) + size);
                new_block->address = (uint32_t)new_block + sizeof(MemoryBlock);
                new_block->size = current->size - size - sizeof(MemoryBlock);
                new_block->is_free = true;
                new_block->next = current->next;
                
                current->size = size;
                current->next = new_block;
            }
            
            current->is_free = false;
            return (void*)current->address;
        }
        current = current->next;
    }
    
    return NULL; // Không đủ bộ nhớ
}

void memory_free(void* ptr) {
    if (!ptr) return;
    
    MemoryBlock* current = memory_list;
    
    // Tìm block cần giải phóng
    while (current != NULL) {
        if ((void*)current->address == ptr) {
            current->is_free = true;
            
            // Gộp các block liền kề
            MemoryBlock* next = current->next;
            if (next && next->is_free) {
                current->size += sizeof(MemoryBlock) + next->size;
                current->next = next->next;
            }
            break;
        }
        current = current->next;
    }
}

MemoryStats get_memory_stats(void) {
    MemoryStats stats = {0};
    MemoryBlock* current = memory_list;
    
    stats.total_memory = total_memory_size;
    
    while (current != NULL) {
        stats.total_blocks++;
        if (current->is_free) {
            stats.free_blocks++;
            stats.free_memory += current->size;
        }
        current = current->next;
    }
    
    stats.used_memory = stats.total_memory - stats.free_memory;
    return stats;
}

void print_memory_stats(void) {
    MemoryStats stats = get_memory_stats();
    printf("Memory Statistics:\n");
    printf("Total Memory: %zu bytes\n", stats.total_memory);
    printf("Used Memory: %zu bytes\n", stats.used_memory);
    printf("Free Memory: %zu bytes\n", stats.free_memory);
    printf("Total Blocks: %u\n", stats.total_blocks);
    printf("Free Blocks: %u\n", stats.free_blocks);
}