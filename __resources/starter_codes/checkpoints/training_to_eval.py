import torch
import sys

checkpoint = torch.load(sys.argv[1])

eval_checkpoint = {'state_dict' : checkpoint['state_dict']}

torch.save(eval_checkpoint, "{}_eval".format(sys.argv[1]))