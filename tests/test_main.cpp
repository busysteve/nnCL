#include <cstdio>
#include <cstring>

int test_shapes();
int test_gradients();
int test_checkpoint();
int test_mnist();

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: nn_tests shapes|grads|ckpt|mnist\n");
    return 1;
  }
  if (std::strcmp(argv[1], "shapes") == 0) return test_shapes();
  if (std::strcmp(argv[1], "grads") == 0) return test_gradients();
  if (std::strcmp(argv[1], "ckpt") == 0) return test_checkpoint();
  if (std::strcmp(argv[1], "mnist") == 0) return test_mnist();
  std::fprintf(stderr, "usage: nn_tests shapes|grads|ckpt|mnist\n");
  return 1;
}
