import torch
import time


M = torch.rand(4096, 2**15).cuda()
v = torch.rand(1, 4096).cuda()

torch.cuda.synchronize()
t_start = time.monotonic_ns()
result  = torch.matmul(v,M).view(-1)
result  = torch.topk(result, 4)
print(result)
torch.cuda.synchronize()
t_stop  = time.monotonic_ns()

elapsed_time = t_stop - t_start


print(elapsed_time*(1e-6))