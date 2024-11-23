#include <stdio.h>
#include "../api/api_core.c"

// Giao diện cơ bản
void show_main_menu() {
    printf("Welcome to UniOS!");
    printf("1. Quản lý bộ nhớ");
    printf("2. Quản lý tiến trình");
    printf("3. Quản lý tệp");
    printf("4. Quản lý sandbox");
    printf("5. Thoát");
}

void manage_sandbox() {
    printf("1. Tạo sandbox");
    printf("2. Kết thúc sandbox");
    printf("3. Quay lại");
    printf("Lựa chọn: ");
    int choice;
    char process_name[50];
    int sandbox_id;

    scanf("%d", &choice);
    switch (choice) {
        case 1:
            printf("Nhập tên tiến trình: ");
            scanf("%s", process_name);
            api_create_sandbox(process_name);
            break;
        case 2:
            printf("Nhập ID sandbox: ");
            scanf("%d", &sandbox_id);
            api_terminate_sandbox(sandbox_id);
            break;
        case 3:
            return;
        default:
            printf("Lựa chọn không hợp lệ.");
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
                printf("Tính năng Quản lý tệp đang phát triển...\n");
                break;
            case 4:
                manage_sandbox();
                break;
            case 5:
                printf("Thoát UniOS.");
                return 0;
            default:
                printf("Lựa chọn không hợp lệ.");
        }
    }
}
