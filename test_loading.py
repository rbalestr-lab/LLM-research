import transformers
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--peft", type=Path, default=None)

    args = parser.parse_args()

    config = transformers.AutoConfig.from_pretrained(args.name)
    if args.pretrained:
        model = transformers.AutoModel.from_pretrained(
            Path("test_checkpoint")
        )  # config=config)
    elif args.peft is not None:
        model.load_adapter(peft_model_id)
    else:
        if "apple" in args.name:
            model = llm_reconstruction_free.modeling_openelm.OpenELMForCausalLM(a)
        else:
            model = transformers.AutoModel.from_config(config)
        params = list(model.parameters())
        print(params[0])
        model.init_weights()
        print(params[0])
        model.post_init()
        print(params[0])
#    model.save_pretrained("test_checkpoint")
