import json
import sys
import pandas as pd
import matplotlib.pyplot as plt

filename1 = sys.argv[1]
filename2 = sys.argv[2]

with open(filename1) as f :
    data1 = [json.loads(d) for d in f]
with open(filename2) as f :
    data2 = [json.loads(d) for d in f]

count1 = dict()
count2 = dict()
for line in range(len(data1)) :
    compress1 = data1[line]['compress']
    compress2 = data2[line]['compress']
    took1 = data1[line]['time(sec)']
    took2 = data2[line]['time(sec)']

    if compress1 not in count1 :
        count1[compress1] = list()
    if compress2 not in count2 :
        count2[compress2] = list()

    count1[compress1].append(took1)
    count2[compress2].append(took2)

    
count1 = dict(sorted(count1.items(), reverse=False))
count2 = dict(sorted(count2.items(), reverse=False))

df_A = {
        "count": count1.keys(), 
        "num_data": [len(count1[x]) for x in count1.keys()], 
        "avg_time": [sum(count1[x]) / len(count1[x]) for x in count1.keys()]
        }
df_A = pd.DataFrame(df_A)

df_B = {
        "count": count2.keys(), 
        "num_data": [len(count2[x]) for x in count2.keys()], 
        "avg_time": [sum(count2[x]) / len(count2[x]) for x in count2.keys()]}
df_B = pd.DataFrame(df_B)


fig, ax1 = plt.subplots(figsize=(10,5))

ax1.bar(df_A["count"] - 0.15, df_A["num_data"], width=0.3, label="2C - Num Data", alpha=0.6)
ax1.bar(df_B["count"] + 0.15, df_B["num_data"], width=0.3, label="3C - Num Data", alpha=0.6)
ax1.set_xlabel("Compress Count")
ax1.set_ylabel("Data #")
ax1.tick_params(axis="y")

# avg_time → line (2차 y축)
ax2 = ax1.twinx()
ax2.plot(df_A["count"], df_A["avg_time"], marker="o", color="tab:blue", label="2C - Avg Time")
ax2.plot(df_B["count"], df_B["avg_time"], marker="s", color="tab:red", label="3C - Avg Time")
ax2.set_ylabel("Average Time (sec)")
ax2.tick_params(axis="y", labelcolor="black")

# --- 범례 정리 ---
lines, labels = [], []
for ax in [ax1, ax2]:
    line, label = ax.get_legend_handles_labels()
    lines += line
    labels += label
ax1.legend(lines, labels, loc="upper right")

plt.title("2C vs 3C")
plt.tight_layout()
plt.xlim(0, 25)

plt.savefig("2c_vs_3c.png", format='png')
