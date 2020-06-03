import matplotlib.pyplot as plt
import sys

assert len(sys.argv) > 1

data = sys.argv[1]

with open(data, 'r') as f:
    lines = f.readlines()

frame_no = []
loss = []
reward = []

for line in lines:
    if "Loss" in line:
        line = line.split(sep=',')
        frame_no.append(int(line[0].split(sep=':')[1].strip()))
        loss.append(float(line[1].split(sep=':')[1].strip())*100)
    elif "reward" in line:
        reward.append(float(line.split(sep=':')[1].strip()))

plt.plot(frame_no, loss, 'bs', frame_no, reward, 'g^')
plt.title(data)
plt.xlabel('Frame number')
plt.show()