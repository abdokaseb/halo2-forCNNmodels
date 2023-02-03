import torch
import numpy as np
np.random.seed(10)
torch.manual_seed(10)
torch.set_printoptions(threshold=51000000)

def write_to_file(name, arr):
  fout = open(name, "w")
  fout.writelines([f"{len(arr.size())}\n"])
  fout.writelines([str(i) + "\n" for i in arr.size()])
  for i in range(len(arr)):
    if len(arr[i].size()):
      for j in range(len(arr[i])):
        if len(arr[i][j].size()):
          for k in range(len(arr[i][j])):
            if len(arr[i][j][k].size()):
              for o in range(len(arr[i][j][k])):
                if len(arr[i][j][k][o].size()):
                  print("err")
                else:
                  fout.writelines([f"{arr[i][j][k][o]}\n"])
            else:
              fout.writelines([f"{arr[i][j][k]}\n"])
        else:
          fout.writelines([f"{arr[i][j]}\n"])
    else:
      fout.writelines([f"{arr[i]}\n"])

def write_other_to_file(file_name, stride, biases, b, Zs, scale_a, scale_b, e):
  fout = open(file_name, "w")
  fout.writelines([f"{stride}\n"])
  fout.writelines([f"{len(biases)}\n"])
  for bias in biases:
    fout.writelines([f"{len(bias)}\n"])
    fout.writelines([str(i.item()) + "\n" for i in bias])
  
  fout.writelines([f"{len(b)}\n"])
  fout.writelines([str(i.item()) + "\n" for i in b])

  fout.writelines([f"{len(Zs)}\n"])
  fout.writelines([str(i.item()) + "\n" for i in Zs])

  fout.writelines([f"{len(scale_a)}\n"])
  fout.writelines([str(i.item()) + "\n" for i in scale_a])
    
  fout.writelines([f"{scale_b}\n"])

  fout.writelines([f"{int(e)}\n"])




bias = [torch.randint(0, 10, (4,)), torch.randint(0, 10, (5,)), torch.randint(0, 10, (2,))]
b = torch.randint(0, 10, (2,))


w = torch.randint(0, 10, (2, 8))
inputs = torch.randint(0, 10, (1, 28, 28))

filters = [torch.randint(0, 10, (4,1, 3, 3)), torch.randint(0, 10, (5,4, 3, 3)), torch.randint(0, 10, (2,5, 3, 3))]

stride = 2

# Zs = torch.randint(0, 8, (len(filters) +1,))
Zs = torch.Tensor([3, 8, 5,30]).int()
# print(Zs)
# scale_as = torch.randint(0, 19, (len(filters) +1,))
scale_as = torch.Tensor([5,7,9,10]).int()
# print(scale_as)
scale_b = 51
  
write_to_file("examples/inputs.txt", inputs)
write_to_file("examples/f0.txt", filters[0])
write_to_file("examples/f1.txt", filters[1])
write_to_file("examples/f2.txt", filters[2])
write_to_file("examples/w1.txt", w)

write_other_to_file("examples/other_parm.txt", stride, bias, b, Zs, scale_as, scale_b, 21312)

conv_in = inputs
for i in range(len(filters)):
  conv_out = torch.nn.functional.conv2d((conv_in - Zs[i]).float(), filters[i].float(), bias = bias[i], padding=0, stride = stride)
  conv_in = torch.nn.functional.relu(conv_out * scale_as[i] / scale_b).int()
  conv_in = torch.minimum(conv_in, torch.Tensor([255.0])).float()


final_out = torch.nn.functional.linear((conv_in.flatten() - Zs[-1]).long(), w, b)

for i in range(len(final_out)):
    print(str(hex(final_out[i])))

write_other_to_file("examples/other_parm.txt", stride, bias, b, Zs, scale_as, scale_b, final_out[0])

# [tensor([7, 5, 2, 7]), tensor([2, 5, 7, 2, 1]), tensor([5, 6])]
# tensor([3, 1])
# torch.Size([8])
# tensor(178087373)
# tensor(278006361)