#include <stdio.h>
#include "../api/api_core.c"

#define MAX_FILES 100 // Định nghĩa số lượng tệp tối đa

struct File {
    char name[50];
    // ... các thuộc tính khác nếu cần ...
};

struct File file_table[MAX_FILES]; // Khai báo mảng tệp

int file_count = 0;

// Giao diện cơ bản
void list_files() {
    printf("\nListing all files:\n");
    if (file_count == 0) {
        printf("No files found.\n");
        return;
    }
    for (int i = 0; i < file_count; i++) {
        printf("File %d: %s\n", i + 1, file_table[i].name);
    }
}

void show_main_menu() {
    printf("\nWelcome to UniOS!\n");
    printf("1. Quản lý bộ nhớ\n");
    printf("2. Quản lý tiến trình\n");
    printf("3. Quản lý tệp\n");
    printf("4. Xem danh sách tệp\n");
    printf("5. Thoát\n");
}

void manage_files() {
    printf("\n1. Tạo tệp\n2. Ghi nội dung vào tệp\n3. Đọc nội dung tệp\n4. Xóa tệp\n5. Quay lại\nLựa chọn: ");
    int choice;
    char filename[50];
    char content[1024];

    scanf("%d", &choice);
    switch (choice) {
        case 1:
            printf("Nhập tên tệp: ");
            scanf("%s", filename);
            create_file(filename);
            break;
        case 2:
            printf("Nhập tên tệp: ");
            scanf("%s", filename);
            printf("Nhập nội dung: ");
            scanf(" %[^\n]", content); // Đọc chuỗi với khoảng trắng
            write_file(filename, content);
            break;
        case 3:
            printf("Nhập tên tệp: ");
            scanf("%s", filename);
            read_file(filename);
            break;
        case 4:
            printf("Nhập tên tệp: ");
            scanf("%s", filename);
            delete_file(filename);
            break;
        case 5:
            return;
        default:
            printf("Lựa chọn không hợp lệ.\n");
    }
}

int main() {
    int choice;
    while (1) {
        show_main_menu();
        printf("Lựa chọn: ");
        scanf("%d", &choice);
        switch (choice) {
            case 1:
                printf("Tính năng Quản lý bộ nhớ đang phát triển...\n");
                break;
            case 2:
                printf("Tính năng Quản lý tiến trình đang phát triển...\n");
                break;
            case 3:
                manage_files();
                break;
            case 4:
                list_files();
                break;
            case 5:
                printf("Thoát UniOS.\n");
                return 0;
            default:
                printf("Lựa chọn không hợp lệ.\n");
        }
    }
}