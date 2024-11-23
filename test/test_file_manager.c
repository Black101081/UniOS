#include "../kernel/file_manager.h"

void test_file_manager() {
    printf("Bắt đầu kiểm thử File Manager...\n");

    // Tạo tệp
    create_file("test1.txt");
    create_file("test2.txt");

    // Ghi và đọc nội dung
    write_file("test1.txt", "Hello, World!");
    read_file("test1.txt");

    // Kiểm tra số lượng file
    printf("Số lượng file hiện tại: %d\n", get_file_count());

    // Xóa tệp
    delete_file("test1.txt");
    read_file("test1.txt"); // Kiểm tra đọc sau khi xóa

    printf("Kiểm thử hoàn thành.\n");
}

int main() {
    test_file_manager();
    return 0;
}