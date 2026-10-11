#include "mini_llama/ops.h"
#include "mini_llama/matmul_dispatch.h"
#include <cmath>
#include <stdexcept>
using namespace mini_lla;
static void nearv(float a,float b){if(std::fabs(a-b)>1e-4f)throw std::runtime_error("value mismatch");}
int test_ops(){Tensor a({2,3});a.data={1,2,3,4,5,6};Tensor b({3,2});b.data={1,2,3,4,5,6};Tensor m=matmul(a,b),n=matmul_dispatch(a,b,MatmulMode::kNaive);nearv(m.at2(0,0),22);nearv(m.at2(1,1),64);for(size_t i=0;i<m.size();++i)nearv(m.data[i],n.data[i]);Tensor x({3});x.data={1,2,3};Tensor w({3});w.data={1,1,1};nearv(rms_norm(x,w).data[0],0.46291f);Tensor p=softmax(x);nearv(p.data[0]+p.data[1]+p.data[2],1);if(argmax(x)!=2)return 1;Tensor q({1,4});q.data={1,0,0,1};Tensor k=q;rope(q,k,0);nearv(q.data[0],1);nearv(q.data[3],1);return 0;}
