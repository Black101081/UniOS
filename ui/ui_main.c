#include <stdio.h>

// Giao diện cơ bản
void show_main_menu() {
    printf("
Welcome to UniOS!
");
    printf("1. Quản lý bộ nhớ
");
    printf("2. Quản lý tiến trình
");
    printf("3. Quản lý tệp
");
    printf("4. Thoát
");
}

void manage_files() {
    printf("
1. Tạo tệp
2. Xóa tệp
3. Quay lại
Lựa chọn: ");
    int choice;
    char filename[50];
    scanf("%d", &choice);
    switch (choice) {
        case 1:
            printf("Nhập tên tệp: ");
            scanf("%s", filename);
            printf("Tạo tệp: %s
", filename);
            break;
        case 2:
            printf("Nhập tên tệp: ");
            scanf("%s", filename);
            printf("Xóa tệp: %s
", filename);
            break;
        case 3:
            return;
        default:
            printf("Lựa chọn không hợp lệ.
");
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
                printf("Tính năng Quản lý bộ nhớ đang phát triển...
");
                break;
            case 2:
                printf("Tính năng Quản lý tiến trình đang phát triển...
");
                break;
            case 3:
                manage_files();
                break;
            case 4:
                printf("Thoát UniOS.
");
                return 0;
            default:
                printf("Lựa chọn không hợp lệ.
");
        }
    }
}
