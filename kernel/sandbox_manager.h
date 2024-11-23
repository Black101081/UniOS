#ifndef SANDBOX_MANAGER_H
#define SANDBOX_MANAGER_H

void create_sandbox(const char *process_name);
void terminate_sandbox(int sandbox_id);

#endif