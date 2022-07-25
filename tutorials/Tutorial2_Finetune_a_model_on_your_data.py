# # Fine-tuning a model on your own data
#
# For many use cases it is sufficient to just use one of the existing public models that were trained on SQuAD or
# other public QA datasets (e.g. Natural Questions).
# However, if you have domain-specific questions, fine-tuning your model on custom examples will very likely boost
# your performance. While this varies by domain, we saw that ~ 2000 examples can easily increase performance by +5-20%.
#
# This tutorial shows you how to fine-tune a pretrained model on your own dataset.

import logging

# We configure how logging messages should be displayed and which log level should be used before importing Haystack.
# Example log message:
# INFO - haystack.utils.preprocessing -  Converting data/tutorial1/218_Olenna_Tyrell.txt
# Default log level in basicConfig is WARNING so the explicit parameter is not necessary but can be changed easily:
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.nodes import FARMReader
from haystack.utils import augment_squad, fetch_archive_from_http

from pathlib import Path


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

    # **Recommendation: Run training on a GPU. To do so change the `use_gpu` arguments below to `True`

    reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True)
    data_dir = "data/squad20"
    # data_dir = "PATH/TO_YOUR/TRAIN_DATA"
    reader.train(data_dir=data_dir, train_filename="dev-v2.0.json", use_gpu=True, n_epochs=1, save_dir="my_model")

    # Saving the model happens automatically at the end of training into the `save_dir` you specified
    # However, you could also save a reader manually again via:
    reader.save(directory="my_model")

    # If you want to load it at a later point, just do:
    new_reader = FARMReader(model_name_or_path="my_model")

    # ## Distill your model
    # In this case, we have used "distilbert-base-uncased" as our base model.
    # This model was trained using a process called distillation.
    # In this process, a bigger model is trained first and is used to train a smaller model which increases its accuracy.
    # This is why "distilbert-base-uncased" can achieve quite competitive performance while being very small.
    #
    # Sometimes, however, you can't use an already distilled model and have to distil it yourself.
    # For this case, haystack has implemented [distillation features](https://haystack.deepset.ai/guides/model-distillation)..
    # distil()


def distil():
    # ### Augmenting your training data
    # To get the most out of model distillation, we recommend increasing the size of your training data by using data augmentation.
    # You can do this by running the [`augment_squad.py` script](https://github.com/deepset-ai/haystack/blob/master/haystack/utils/augment_squad.py):

    doc_dir = "data/tutorial2"
    # Downloading smaller glove vector file (only for demonstration purposes)
    glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    fetch_archive_from_http(url=glove_url, output_dir=doc_dir)

    # Downloading very small dataset to make tutorial faster (please use a bigger dataset in real use cases)
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/squad_small.json.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # Just replace squad_small.json with the name of your dataset and adjust the output path
    augment_squad.main(
        squad_path=Path("squad_small.json"),
        output_path=Path("augmented_dataset.json"),
        multiplication_factor=2,
        glove_path=Path("glove.6B.300d.txt"),
    )
    # In this case, we use a multiplication factor of 2 to keep this example lightweight.
    # Usually you would use a factor like 20 depending on the size of your training data.
    # Augmenting this small dataset with a multiplication factor of 2, should take about 5 to 10 minutes to run on one V100 GPU.

    # ### Running distillation
    # Distillation in haystack is done in two steps:
    # First, you run intermediate layer distillation on the augmented dataset to ensure the two models behave similarly.
    # After that, you run the prediction layer distillation on the non-augmented dataset to optimize the model for your specific task.

    # If you want, you can leave out the intermediate layer distillation step and only run the prediction layer distillation.
    # This way you also do not need to perform data augmentation. However, this will make the model significantly less accurate.

    # Loading a fine-tuned model as teacher e.g. "deepset/​bert-​base-​uncased-​squad2"
    teacher = FARMReader(model_name_or_path="huawei-noah/TinyBERT_General_6L_768D", use_gpu=True)

    # You can use any pre-trained language model as teacher that uses the same tokenizer as the teacher model.
    # The number of the layers in the teacher model also needs to be a multiple of the number of the layers in the student.
    student = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True)

    student.distil_intermediate_layers_from(
        teacher, data_dir="data/squad20", train_filename="augmented_dataset.json", use_gpu=True
    )
    student.distil_prediction_layer_from(teacher, data_dir="data/squad20", train_filename="dev-v2.0.json", use_gpu=True)

    student.save(directory="my_distilled_model")


if __name__ == "__main__":
    tutorial2_finetune_a_model_on_your_data()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
