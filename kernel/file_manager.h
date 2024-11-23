#ifndef FILE_MANAGER_H
#define FILE_MANAGER_H

#include <stdio.h>

// Định nghĩa cấu trúc File
typedef struct File {
    char name[50];      // Tên file
    char content[1024]; // Nội dung file
    int is_open;        // Trạng thái mở file
} File;

// Khai báo các biến toàn cục
extern File file_table[10];
extern int file_count;

// Khai báo các hàm
void create_file(const char *name);
void write_file(const char *name, const char *content);
void read_file(const char *name);
void delete_file(const char *name);

#endif