#include <stdio.h>
#include <string.h>

typedef struct {
    char process_name[50];
    int sandbox_id;
    int is_active;
} Sandbox;

Sandbox sandbox_table[10];
int sandbox_count = 0;

// Hàm tạo sandbox
void create_sandbox(const char *process_name) {
    if (sandbox_count >= 10) {
        printf("Cannot create sandbox: Sandbox table full.\n");
        return;
    }

    Sandbox new_sandbox; // Khai báo biến
    strncpy(new_sandbox.process_name, process_name, 50);
    new_sandbox.sandbox_id = sandbox_count + 1;
    new_sandbox.is_active = 1;

    sandbox_table[sandbox_count] = new_sandbox;
    sandbox_count++;

    printf("Sandbox created for process '%s' with ID %d.\n", process_name, new_sandbox.sandbox_id);
}

// Hàm kết thúc sandbox
void terminate_sandbox(int sandbox_id) {
    for (int i = 0; i < sandbox_count; i++) {
        if (sandbox_table[i].sandbox_id == sandbox_id && sandbox_table[i].is_active) {
            sandbox_table[i].is_active = 0;
            printf("Sandbox with ID %d terminated.\n", sandbox_id);
            return;
        }
    }
    printf("Sandbox with ID %d not found or already terminated.\n", sandbox_id);
}