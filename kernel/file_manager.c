#include <stdio.h>
#include <string.h>

typedef struct {
    char name[50];      // Tên file
    char content[1024]; // Nội dung file
    int is_open;        // Trạng thái mở file
} File;

File file_table[10];    // Bảng lưu trữ file
int file_count = 0;     // Số lượng file hiện tại

// Hàm tạo file
void create_file(const char *name) {
    if (file_count >= 10) {
        printf("Cannot create file: File table full.\n");
        return;
    }

    File new_file;
    strncpy(new_file.name, name, 50);
    new_file.content[0] = '\0'; // Khởi tạo nội dung trống
    new_file.is_open = 0;

    file_table[file_count] = new_file;
    file_count++;

    printf("File '%s' created.\n", name);
}

// Hàm ghi nội dung vào file
void write_file(const char *name, const char *content) {
    for (int i = 0; i < file_count; i++) {
        if (strcmp(file_table[i].name, name) == 0) {
            strncpy(file_table[i].content, content, 1024);
            printf("Content written to file '%s'.\n", name);
            return;
        }
    }
    printf("File '%s' not found.\n", name);
}

// Hàm đọc nội dung file
void read_file(const char *name) {
    for (int i = 0; i < file_count; i++) {
        if (strcmp(file_table[i].name, name) == 0) {
            printf("Content of file '%s': %s\n", name, file_table[i].content);
            return;
        }
    }
    printf("File '%s' not found.\n", name);
}

// Hàm xóa file
void delete_file(const char *name) {
    for (int i = 0; i < file_count; i++) {
        if (strcmp(file_table[i].name, name) == 0) {
            for (int j = i; j < file_count - 1; j++) {
                file_table[j] = file_table[j + 1];
            }
            file_count--;
            printf("File '%s' deleted.\n", name);
            return;
        }
    }
    printf("File '%s' not found.\n", name);
}