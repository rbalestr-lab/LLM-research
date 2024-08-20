import wandb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 340)
pd.set_option("display.max_rows", 340)

df = pd.DataFrame(columns=["dataset", "arch", "pct", "acc"])
api = wandb.Api()

runs = api.runs("llm-reconstruction-free/8k_corrected_finetuning")
all_data = []
scatter_x = {}
scatter_y_with = {}
scatter_y_without = {}

model = []
comparison_with = {}
comparison_without = {}
for run in runs:
#    if pd.to_datetime(run.metadata["startedAt"]) < pd.to_datetime("07/02/2024"):
#        break
    if run.metadata["gpu_devices"] is None:
        continue
#    if "H100" not in run.metadata["gpu_devices"][0]["name"]:
#        continue
    if run.config["pretrained"]==0 and run.config["vocab_size"] != 32000:
        continue
    
    data = []
    for row in run.scan_history():
        if "eval/accuracy" not in row:
            print(f"No eval/accuracy in {run}")
            print(row)
            break
        data.append(
            [
                row["eval/accuracy"],
                row["eval/balanced_accuracy"],
                row["eval/F1"],
                row["eval/loss"],
                row["train/loss"],
            ]
        )
    if len(data) == 0:
        continue
    timeseries = pd.DataFrame(
        np.asarray(data),
        columns=["accuracy", "balanced_accuracy", "F1", "eval/loss", "train/loss"],
    )
    timeseries["train/loss"] = timeseries["train/loss"].fillna(method="ffill")

    scatter = timeseries["F1"].dropna().astype(float) * 100

    timeseries = (timeseries.fillna(method="ffill") * 100).round(2)
    all_data.append(timeseries.iloc[timeseries["F1"].argmax()])
    if run.config["lora_rank"]:
        ft = f"LoRA({run.config['lora_rank']})"
    elif run.config["freeze"]:
        assert run.config["lora_rank"] == 0
        ft = "None"
    elif run.config["freeze"] == 0:
        ft = "full"
        if run.config["pretrained"]:
            comparison_with[f"{run.config['dataset']}_{run.config['backbone']}"] = (
                timeseries["train/loss"]
            )

            scatter_y_with[f"{run.config['dataset']} {run.config['backbone']}"] = (
                scatter.to_list()
            )
        else:
            comparison_without[
                f"{run.config['dataset']}_{run.config['backbone']}"
            ] = timeseries["train/loss"]
            scatter_y_without[
                f"{run.config['dataset']} {run.config['backbone']}"
            ] = scatter.to_list()

        scatter_x[f"{run.config['dataset']} {run.config['backbone']}"] = list(
            np.linspace(0, 1, len(scatter)) * 100
        )
        model.extend([run.config["backbone"]] * len(scatter))
        #print(scatter_y_with, scatter_y_without)
    else:
        raise RuntimeError()

    try:
        all_data[-1]["dataset"] = str(run.config["dataset"]).replace("_", "-")
        all_data[-1]["backbone"] = str(run.config["backbone"]).replace("_", "-")
        all_data[-1]["steps"] = str(run.config["max_steps"])
        all_data[-1]["finetuning"] = ft
        all_data[-1]["init"] = "pretrained" if run.config["pretrained"] else "random"
    except Exception as e:
        print(e)
        raise(e)
        continue

    print(all_data[-1])



# fig, ax = plt.subplots(1,1, figsize=(9,9))
# for name in comparison_with:
#    if "OpenELM-1-1B" not in name:
#        continue
#    ax.plot(comparison_with[name], color="blue")
#    ax.plot(comparison_without[name], color="black")
# plt.tight_layout()
# plt.savefig("comparison_dynamics.pdf")
# plt.close()


#sns.set(font_scale=2)
#fig, axs = plt.subplots(1, 3, figsize=(18, 6))
#i = 0
#for name in scatter_y_with:
#    if "OpenELM-1_1B" not in name:
#        continue
#    if "bios" in name:
#        continue
#    if "wiki" in name:
#        continue
#    axs[i].plot(scatter_x[name], scatter_y_with[name], color="green")
#    axs[i].plot(scatter_x[name], scatter_y_without[name], color="blue")
#    axs[i].set_title(name.split(" ")[0])
#    axs[i].set_xlabel("% training epochs")
#    i += 1
#axs[0].set_ylabel("test F1 score")
# df = pd.DataFrame(np.stack([scatter_x, scatter_y, model], 1), columns = ["% training steps", "eval F1", "model"])
# df = df.astype({"eval F1":float, "% training steps":float})
# print(df)
# sns.lineplot(df, x="% training steps", y="eval F1", hue="model", ax=ax, markers=True, dashes=False)
# lims = [
#    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
# ]
#
## now plot both limits against eachother
# ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, linewidth=3)
# ax.set_aspect('equal')

#plt.tight_layout()
#plt.savefig("scatter_train_test.pdf")
#plt.close()

df = pd.concat(all_data, axis=1).T

df = pd.pivot_table(
    df,
    values="F1",
    columns=["dataset"],
    index=["backbone", "init", "finetuning"],
    aggfunc='max'#lambda x:len(x),
)
df = df.drop(columns="medical")
df = df.assign(mean=df.mean(1))
print(df.to_latex(float_format="%.1f"))
print(df.groupby(level=2).apply(max).to_latex(float_format="%.1f"))

adf
df = pd.pivot_table(
    df,
    values="F1",
    index=["dataset", "backbone"],
    columns=["init", "finetuning", "steps"],
    aggfunc='max'#lambda x:len(x),
)
print(df)
print(df.loc[:,(slice(None), slice(None), "8000")])
asdf
df = df.loc[
    (
        slice(None),
        [
            "microsoft/phi-2",
            "apple/OpenELM-1-1B",
            "apple/OpenELM-450M",
            "apple/OpenELM-270M",
        ],
    ),
    :,
].sort_index(level=0)
cols = np.asarray(df.columns)
cols[[0, 1, 2, 3]] = cols[[2, 1, 0, 3]]
df.columns = pd.MultiIndex.from_tuples(cols)
df.columns.names = ["init", "finetuning"]
df.iloc[:, [0, 1, 2, 3]] = df.iloc[:, [2, 1, 0, 3]]
print(df.astype(str).to_latex(multicolumn_format="c"))
asdf


asdf
df = pd.concat(data, axis=1).T
df = (
    pd.pivot_table(df, columns="pct", index=["norm", "share", "dataset"], values="acc")
    * 100
)
print(df.to_latex())

for name, group in df.groupby(["dataset", "share"]):
    print(group.sort_values(by="pct"))
