#include <stdio.h>
#include <string.h>

typedef struct {
    char name[50];
    int is_open;
} File;

File file_table[10];
int file_count = 0;

void create_file(const char *name) {
    if (file_count >= 10) {
        printf("Cannot create file: File table full.
");
        return;
    }

    File new_file;
    strncpy(new_file.name, name, 50);
    new_file.is_open = 0;

    file_table[file_count] = new_file;
    file_count++;

    printf("File %s created.
", name);
}

void delete_file(const char *name) {
    for (int i = 0; i < file_count; i++) {
        if (strcmp(file_table[i].name, name) == 0) {
            for (int j = i; j < file_count - 1; j++) {
                file_table[j] = file_table[j + 1];
            }
            file_count--;
            printf("File %s deleted.
", name);
            return;
        }
    }
    printf("File %s not found.
", name);
}
