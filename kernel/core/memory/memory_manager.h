#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Định nghĩa cấu trúc Memory Block
typedef struct MemoryBlock {
    uint32_t address;
    size_t size;
    bool is_free;
    struct MemoryBlock* next;
} MemoryBlock;

// Khai báo các hàm quản lý bộ nhớ
void init_memory_manager(void* start, size_t size);
void* memory_alloc(size_t size);
void memory_free(void* ptr);
void print_memory_stats(void);

// Thống kê bộ nhớ
typedef struct {
    size_t total_memory;
    size_t used_memory;
    size_t free_memory;
    uint32_t total_blocks;
    uint32_t free_blocks;
} MemoryStats;

MemoryStats get_memory_stats(void);

#endif // MEMORY_MANAGER_H