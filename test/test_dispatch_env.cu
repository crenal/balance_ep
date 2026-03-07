#include "test_dispatch_env.h"

#include <stdlib.h>

int read_env_int(const char *name, int *out) {
  const char *value = getenv(name);
  if (!value || value[0] == '\0') return 0;
  *out = atoi(value);
  return 1;
}

int read_env_float(const char *name, float *out) {
  const char *value = getenv(name);
  if (!value || value[0] == '\0') return 0;
  *out = strtof(value, nullptr);
  return 1;
}
