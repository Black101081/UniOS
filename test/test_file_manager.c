#include "../kernel/file_manager.c"

void test_file_manager() {
    // Test tạo file
    create_file("test1.txt");
    create_file("test2.txt");

    // Test ghi và đọc file
    write_file("test1.txt", "Hello, World!");
    read_file("test1.txt");

    // Test xóa file
    delete_file("test1.txt");
    read_file("test1.txt"); // Kiểm tra file đã bị xóa

    // Danh sách file sau khi xóa
    for (int i = 0; i < file_count; i++) {
        printf("Remaining file: %s\n", file_table[i].name);
    }
}

int main() {
    printf("Running tests for File Manager...\n");
    test_file_manager();
    printf("Tests completed.\n");
    return 0;
}