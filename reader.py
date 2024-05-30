import wandb
import pandas as pd
import numpy as np


df = pd.DataFrame(columns=["dataset", "arch", "pct", "acc"])
api = wandb.Api()

runs = api.runs("llm-reconstruction-free/supervised_finetuning")
all_data = []
for run in runs:
    if run.state != "finished":
        continue
    try:
        print(run.state, run.config["backbone"], run.config["dataset"], 
                run.config["freeze"], run.config["lora_rank"], run.config["pretrained"])
        if "L4" in run.metadata["gpu_devices"][0]["name"]:
            continue
        data = []
        for i, row in run.history().iterrows():
            data.append([
                row["eval/accuracy"],
                row["eval/balanced_accuracy"],
                row["eval/F1"]
                ])
        timeseries = pd.DataFrame(np.asarray(data), columns=["accuracy", "balanced_accuracy", "F1"])
        timeseries = (timeseries.dropna().reset_index() * 100).round(2)
        all_data.append(timeseries.iloc[-1])
        init = "pretrained" if run.config["pretrained"] else "random"
        if run.config["lora_rank"]:
            ft = f"LoRA({run.config['lora_rank']})"
        elif run.config["freeze"]:
            assert run.config['lora_rank'] == 0
            ft = "None"
        elif run.config["freeze"] == 0:
            ft = "full"
        else:
            raise RuntimeError()

        all_data[-1]["dataset"] = str(run.config["dataset"]).replace("_", "-")
        all_data[-1]["backbone"] = str(run.config["backbone"]).replace("_", "-")
        all_data[-1]["finetuning"] = ft
        all_data[-1]["init"] = init


    except Exception as e:
        print(e)
    continue

df = pd.concat(all_data, axis=1).T
df = pd.pivot_table(df, values="F1", index=["dataset", "backbone"], columns=["init", "finetuning"], aggfunc="mean")
cols = np.asarray(df.columns)
cols[[0,1,2,3]] = cols[[2,1,0,3]]
df.columns = pd.MultiIndex.from_tuples(cols)
df.columns.names = ["init", "finetuning"]
df.iloc[:,[0, 1, 2, 3]] = df.iloc[:,[2, 1, 0, 3]]
print(df.astype(str).to_latex(multicolumn_format="c"))
asdf



asdf
df = pd.concat(data, axis=1).T
df = pd.pivot_table(df, columns="pct", index=["norm", "share", "dataset"], values="acc") * 100
print(df.to_latex())

for name, group in df.groupby(["dataset", "share"]):
    print(group.sort_values(by="pct"))
