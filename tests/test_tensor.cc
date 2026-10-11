#include "mini_llama/tensor.h"
using namespace mini_lla;
int test_tensor(){Tensor t({2,3});t.at2(1,2)=5;if(t.size()!=6||t.at2(1,2)!=5)return 1;if(t.reshape({3,2}).size()!=6)return 1;return 0;}
