# # Fine-tuning a model on your own data
# 
# For many use cases it is sufficient to just use one of the existing public models that were trained on SQuAD or
# other public QA datasets (e.g. Natural Questions).
# However, if you have domain-specific questions, fine-tuning your model on custom examples will very likely boost
# your performance. While this varies by domain, we saw that ~ 2000 examples can easily increase performance by +5-20%.
# 
# This tutorial shows you how to fine-tune a pretrained model on your own dataset.

from haystack.reader.farm import FARMReader


def tutorial2_finetune_a_model_on_your_data():
    # ## Create Training Data
    #
    # There are two ways to generate training data
    #
    # 1. **Annotation**: You can use the annotation tool(https://github.com/deepset-ai/haystack#labeling-tool) to label
    #                    your data, i.e. highlighting answers to your questions in a document. The tool supports structuring
    #                   your workflow with organizations, projects, and users. The labels can be exported in SQuAD format
    #                    that is compatible for training with Haystack.
    #
    # 2. **Feedback**:   For production systems, you can collect training data from direct user feedback via Haystack's
    #                    REST API interface. This includes a customizable user feedback API for providing feedback on the
    #                    answer returned by the API. The API provides a feedback export endpoint to obtain the feedback data
    #                    for fine-tuning your model further.
    #
    #
    # ## Fine-tune your model
    #
    # Once you have collected training data, you can fine-tune your base models.
    # We initialize a reader as a base model and fine-tune it on our own custom dataset (should be in SQuAD-like format).
    # We recommend using a base model that was trained on SQuAD or a similar QA dataset before to benefit from Transfer
    # Learning effects.

    #**Recommendation: Run training on a GPU. To do so change the `use_gpu` arguments below to `True`

    reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True)
    train_data = "data/squad20"
    # train_data = "PATH/TO_YOUR/TRAIN_DATA"
    reader.train(data_dir=train_data, train_filename="dev-v2.0.json", use_gpu=True, n_epochs=1, save_dir="my_model")

    # Saving the model happens automatically at the end of training into the `save_dir` you specified
    # However, you could also save a reader manually again via:
    reader.save(directory="my_model")

    # If you want to load it at a later point, just do:
    new_reader = FARMReader(model_name_or_path="my_model")


if __name__ == "__main__":
    tutorial2_finetune_a_model_on_your_data()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/