#ifndef PROCESS_MANAGER_H
#define PROCESS_MANAGER_H

#include <stdint.h>

typedef struct {
    uint32_t pid;
    char name[256];
    uint8_t priority;
    uint8_t state;
} Process;

extern int create_process(const char* name, uint8_t priority);
extern int terminate_process(uint32_t pid);
extern Process* get_process(uint32_t pid);
extern void list_processes(void);

#endif
