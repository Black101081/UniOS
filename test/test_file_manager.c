#include "../kernel/file_manager.c"

void test_file_manager() {
    create_file("test1.txt");
    write_file("test1.txt", "Hello, World!");
    read_file("test1.txt");
    delete_file("test1.txt");
}

int main() {
    printf("Running tests for File Manager...");
    test_file_manager();
    printf("Tests completed.");
    return 0;
}
