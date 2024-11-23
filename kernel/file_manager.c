#include "file_manager.h"

File file_table[10];    // Bảng lưu trữ file
int _file_count = 0;     // Số lượng file hiện tại

int get_file_count() {
    return _file_count;
}

// Các hàm quản lý file
void create_file(const char *name) {
    if (_file_count >= 10) {
        printf("Cannot create file: File table full.\n");
        return;
    }
    strncpy(file_table[_file_count].name, name, 50);
    file_table[_file_count].is_open = 0;
    file_table[_file_count].content[0] = '\0';
    _file_count++;
    printf("File '%s' created.\n", name);
}

void write_file(const char *name, const char *content) {
    for (int i = 0; i < _file_count; i++) {
        if (strcmp(file_table[i].name, name) == 0) {
            strncpy(file_table[i].content, content, 1024);
            printf("Content written to file '%s'.\n", name);
            return;
        }
    }
    printf("File '%s' not found.\n", name);
}

void read_file(const char *name) {
    for (int i = 0; i < _file_count; i++) {
        if (strcmp(file_table[i].name, name) == 0) {
            printf("Content of file '%s': %s\n", name, file_table[i].content);
            return;
        }
    }
    printf("File '%s' not found.\n", name);
}

void delete_file(const char *name) {
    for (int i = 0; i < _file_count; i++) {
        if (strcmp(file_table[i].name, name) == 0) {
            for (int j = i; j < _file_count - 1; j++) {
                file_table[j] = file_table[j + 1];
            }
            _file_count--;
            printf("File '%s' deleted.\n", name);
            return;
        }
    }
    printf("File '%s' not found.\n", name);
}