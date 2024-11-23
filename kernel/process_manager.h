#ifndef PROCESS_MANAGER_H
#define PROCESS_MANAGER_H

#include <stdio.h>
#include <string.h>

// Định nghĩa cấu trúc tiến trình
typedef struct {
    int pid;              // ID của tiến trình
    char name[50];        // Tên tiến trình
    int is_active;        // Trạng thái hoạt động
} Process;

// Khai báo các biến toàn cục
extern Process process_table[10]; // Chỉ khai báo, không định nghĩa
extern int process_count;         // Chỉ khai báo, không định nghĩa

// Khai báo các hàm
void create_process(const char *name);
void terminate_process(int pid);
void list_processes();

#endif