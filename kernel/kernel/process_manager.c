#include <stdio.h>

// Quản lý tiến trình đơn giản
void create_process(const char *name) {
    printf("Process %s created.\n", name);
}

void terminate_process(const char *name) {
    printf("Process %s terminated.\n", name);
}
