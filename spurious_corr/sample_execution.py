import random
from .transform import spurious_transform
from .modifiers import CompositeModifier, ItemInjection, HTMLInjection
from .generators import spurious_date_generator
from .utils import pretty_print, pretty_print_dataset, highlight_dates, highlight_from_file, highlight_html
import llm_research.data

def main():
    # dataset_name = "imdb"
    # dataset_name = "imdb"
    # data = llm_research.data.from_name(dataset_name)
    # train_dataset, test_dataset = data["train"], data["test"]
    
    # Create an injection modifier
    # modifier = ItemInjection.from_function(
    #     spurious_date_generator, location="random", token_proportion=0.2
    # )

    # modifier = ItemInjection.from_file(
    #     "spurious_corr/data/colors.txt", location="random", token_proportion=0.2
    # )

    # text = "<a>asdf <b> asdf sadf asdf </b> asdf asdf </a>"
    # text = "sample piece of text"


    text = "<a>asdf <b> asdf sadf asdf </b> asdf asdf </a>"
    modifier = HTMLInjection.from_list(["<p> </p>"], location="end", level=2)

    # # # modifier_1 = HTMLInjection.from_file("spurious_corr/data/html_tags.txt", location="end")

    # text = "sample piece of text"
    # m1 = HTMLInjection.from_list(["<A> </A>"], location="end", level=0)
    # m2 = HTMLInjection.from_list(["<B> </B>"], location="end", level=1)
    # m3 = HTMLInjection.from_list(["<C> </C>"], location="end", level=2)
    # m4 = HTMLInjection.from_list(["<D> </D>"], location="end", level=3)
    # m5 = HTMLInjection.from_list(["<E> </E>"], location="end", level=2)

    # modifier = CompositeModifier([m1, m2, m3, m4, m5])
    text = modifier(text)
    
    # Apply the transformation to all examples with label 1 in the training dataset.
    # modified_data = spurious_transform(
    #     label_to_modify=1,
    #     dataset=train_dataset,
    #     modifier=modifier,
    #     text_proportion=1.0,
    # )

    # highlight_func = highlight_from_file("spurious_corr/data/colors.txt")
    highlight_func = highlight_html("spurious_corr/data/html_tags.txt")

    pretty_print(text, highlight_func)
    
    # Print out the first three modified examples for label 1 with date highlighting.
    # pretty_print_dataset(modified_data, n=3, highlight_func=highlight_func, label=1)

if __name__ == '__main__':
    main()