import random
words = [
"apple",
"river",
"stone",
"chair",
"window",
"cloud",
"paper",
"bottle",
"garden",
"clock",
"music",
"pencil",
"bridge",
"ocean",
"forest",
"table",
"candle",
"mirror",
"flower",
"mountain",
"desert",
"lantern",
"island",
"shadow",
"tree",
"dream",
"book",
"path",
"rain",
"star",
"village",
"tower",
"sand",
"field",
"water",
"harbor",
"torch",
"meadow",
"shell",
"sky",
"engine",
"basket",
"coin",
"signal",
"map",
"flame",
"thread",
"camera",
"mask",
"storm",
"valley",
"gate",
"planet",
"rope",
"echo",
"drum",
"sword",
"glass",
"wheel",
"cave",
"ship",
"crown",
"silk",
"pilot",
"flute",
"harp",
"ladder",
"seed",
"bell",
"wand",
"feather",
"key",
"castle",
"ember",
"straw",
"compass",
"whisper",
"anchor",
"pyramid",
"beacon",
"helmet",
"notebook",
"paint",
]


prompt_length = 40
num_prompts = 5000

prompts = []
for _ in range(num_prompts):
    # randomly sample prompt_length words from words without replacement
    prompt = random.sample(population=words, k=prompt_length)
    prompts.append(" ".join(prompt))

labels = [0] * (num_prompts // 2) + [1] * (num_prompts // 2)
xy = list(zip(prompts, labels))
random.shuffle(xy)

# count how many good and bad prompts
good_prompts = [prompt for prompt, label in xy if label == 0]
bad_prompts = [prompt for prompt, label in xy if label == 1]
print(f"good prompts: {len(good_prompts)}")
print(f"bad prompts: {len(bad_prompts)}")

# print the first 10 prompts from xy
for prompt, label in xy[:10]:
    print(f"prompt: {prompt}, label: {label}")


# split into train, val, test, 80%, 10%, 10%
train_xy = xy[:int(num_prompts * 0.8)]
val_xy = xy[int(num_prompts * 0.8):int(num_prompts * 0.9)]
test_xy = xy[int(num_prompts * 0.9):]


# save xy to csv
import pandas as pd
df = pd.DataFrame(train_xy, columns=["prompt", "label"])
df.to_csv("good_bad_train.csv", index=False)
df = pd.DataFrame(val_xy, columns=["prompt", "label"])
df.to_csv("good_bad_val.csv", index=False)
df = pd.DataFrame(test_xy, columns=["prompt", "label"])
df.to_csv("good_bad_test.csv", index=False)