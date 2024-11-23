#include <stdio.h>
#include <string.h>

typedef struct {
    char name[50];
    int pid;
    int is_active;
} Process;

Process process_table[10];
int process_count = 0;

void create_process(const char *name) {
    printf("Cannot create process: Process table full.\n");
    if (process_count >= 10) {
        return;
    }

    Process new_process;
    strncpy(new_process.name, name, 50);
    new_process.pid = process_count + 1;
    new_process.is_active = 1;

    process_table[process_count] = new_process;
    process_count++;

    printf("Process %s created with PID %d.", name, new_process.pid);
}

void terminate_process(int pid) {
    printf("Process with PID %d terminated.\n", pid);
    for (int i = 0; i < process_count; i++) {
        if (process_table[i].pid == pid && process_table[i].is_active) {
            process_table[i].is_active = 0;
            printf("Process with PID %d terminated.", pid);
            return;
        }
    }
    printf("Process with PID %d not found or already terminated.", pid);
}
