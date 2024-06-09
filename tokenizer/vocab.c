#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void write_combination(FILE *file, const char *combination, int index, int first) {
  if (!first) {
    fprintf(file, ",\n");
  }
  fprintf(file, "\"");

  for (int i = 0; combination[i] != '\0'; i++) {
    if (combination[i] == '\n') {
      fprintf(file, "\\n");
    } else {
      fprintf(file, "%c", combination[i]);
    }
  }
  fprintf(file, "\": %d", index);
}

void generate_combinations(const char *data_str, int data_len, int n_str, FILE *file) {
  int *indices = (int *)malloc(n_str * sizeof(int));
  if (!indices) {
    fprintf(stderr, "Memory allocation failed\n");
    return;
  }

  char *combination = (char *)malloc((n_str + 1) * sizeof(char));
  combination[n_str] = '\0';

  for (int i = 0; i < n_str; ++i) {
    indices[i] = 0;
  }

  fprintf(file, "{\n");

  int first = 1;
  int index = 0;
  while (1) {
    for (int i = 0; i < n_str; ++i) {
      combination[i] = data_str[indices[i]];
    }

    write_combination(file, combination, index++, first);
    first = 0;

    int i;
    for (i = n_str - 1; i >= 0; --i) {
      if (indices[i] < data_len - 1) {
        indices[i]++;
        break;
      }
      indices[i] = 0;
    }

      if (i < 0) {
        break;
      }
    }

    fprintf(file, "\n}\n");

    free(indices);
    free(combination);
}

int main() {
  const char data_str[] = {'a', 't', 'c', 'g', '\n'};
  int n_str = 5;
  int data_len = sizeof(data_str) / sizeof(data_str[0]);

  FILE *file = fopen("vocab.json", "w");
  if (!file) {
    fprintf(stderr, "Could not open file for writing\n");
    return 1;
  }

  generate_combinations(data_str, data_len, n_str, file);

  fclose(file);
  printf("Written to the file!\n");

  return 0;
}