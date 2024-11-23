#ifndef FILE_MANAGER_H
#define FILE_MANAGER_H

#include <stdio.h>
#include <string.h>

typedef struct File{
    char name[50];      // Tên file
    char content[1024]; // Nội dung file
    int is_open;        // Trạng thái mở file
} File;

extern File file_table[10]; // Bảng lưu trữ file
extern int _file_count;     // Số lượng file hiện tại

int get_file_count();       // Hàm lấy số lượng file

void create_file(const char *name);
void write_file(const char *name, const char *content);
void read_file(const char *name);
void delete_file(const char *name);

#endif