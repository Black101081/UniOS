#include "process_manager.h"

Process process_table[10]; // Định nghĩa biến toàn cục ở đây
int process_count = 0;     // Định nghĩa biến toàn cục ở đây

void create_process(const char *name) {
    if (process_count >= 10) {
        printf("Cannot create process: Process table full.\n");
        return;
    }
    Process new_process;
    new_process.pid = process_count + 1;
    strncpy(new_process.name, name, 50);
    new_process.is_active = 1;

    process_table[process_count] = new_process;
    process_count++;

    printf("Process '%s' created with PID %d.\n", name, new_process.pid);
}

void terminate_process(int pid) {
    for (int i = 0; i < process_count; i++) {
        if (process_table[i].pid == pid && process_table[i].is_active) {
            process_table[i].is_active = 0;
            printf("Process with PID %d terminated.\n", pid);
            return;
        }
    }
    printf("Process with PID %d not found or already terminated.\n", pid);
}

void list_processes() {
    printf("Listing all active processes:\n");
    for (int i = 0; i < process_count; i++) {
        if (process_table[i].is_active) {
            printf("PID: %d, Name: %s\n", process_table[i].pid, process_table[i].name);
        }
    }
}