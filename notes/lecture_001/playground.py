import torch

def time_pytorch_function(func, input):
  # CUDA IS ASYNC so can't use python time module
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)

  # Warmup
  for _ in range(5):
    func(input)

  start.record()
  func(input)
  end.record()
  torch.cuda.synchronize()
  return start.elapsed_time(end)

def warmup(func, input):
  for _ in range(5):
    func(input)

x = torch.randn(10000, 10000).cuda()

def profile(func):
  warmup(func, x)
  with torch.profiler.profile() as prof:
    func(x)
  print("=============")
  print(f"Profiling {func.__name__}")
  print("=============")
  print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

profile(torch.square)

def square2(x):
  return x ** 2

profile(square2)


def square3(x):
  return x * x

profile(square3)


## TODO: learn how to write an inline square matrix in cuda, then load to inline pytorch, then profile
