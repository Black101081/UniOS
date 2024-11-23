#include <stdio.h>

// Giao diện cơ bản
void show_main_menu() {
    printf("Welcome to UniOS!\n");
    printf("1. Quản lý bộ nhớ\n");
    printf("2. Quản lý tiến trình\n");
    printf("3. Thoát\n");
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
                printf("Thoát UniOS.\n");
                return 0;
            default:
                printf("Lựa chọn không hợp lệ.\n");
        }
    }
}
