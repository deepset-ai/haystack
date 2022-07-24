
"""
this is very minimal implementation,
it should be extended to include model card for all of the model.
Also, suggested to add user prompt to review model card
"""

# METADATA
# ---
# language:
#   - "List of ISO 639-1 code for your language"
#   - lang1
#   - lang2
# thumbnail: "url to a thumbnail used in social sharing"
# tags:
# - tag1
# - tag2
# license: "any valid license identifier"
# datasets:
# - dataset1
# - dataset2
# metrics:
# - metric1
# - metric2
# ---


import warnings
import yaml

def extract_parameters(model,transformer_models, trainer=None):
  """
  compact module to extract model information required by MODEL_CARD initialization

  """
  model_details = dict()
  def extract_training_details(trainer):
      """
      extracts learning rate, optimizer, seed, device , etc
      and return the dictionary of training parameters

      """
      params = dict()
      return params

  def extract_evaluation_details(trainer):
      """
      extracts evaluation dataset, batch size, results, metrics and so on
      """
      params = dict()
      return params
  others = dict()
  others["version"] = transformer_models._version
  others["device"] = transformer_models.device.type
  try:
      others["number of labels"]= transformer_models.num_labels
  except:
      others["number of labels"]= "not found"
  others["number_of_parameters"] = sum(p.numel() for p in transformer_models.parameters())
  others["language"] = "en"
  try:
      others["base_model"] = model.model_name_or_path
  except:
      others["base_model"] = "this is a base model itself"

  model_details["model_details"] = others
  model_details["model_description"] = model.__doc__
  if isinstance(model.type, str):
      model_details["model_type"] = model.type
  else:
      model_details["model_type"] = "not defined"


  if trainer is not None:
    model_details["training_data"] = extract_training_details()
    model_details["evaluation_data"] =  extract_evaluation_details()

  return model_details


MODEL_CARD_NAME = "README.md"
_MODEL_METRIC = {

    "FARMReader": ["BLEU"]
}

## this is temporary, one should extract all of these values from either model itself or by requesting input from user

_DEFAULT_TAGS = {
    "FARMReader":["questionanswering"]

}


class ModelCard:
    r"""
    Class to define structured model card. Refers the Structure from https://huggingface.co/docs/hub/models-cards
    based on the paper https://arxiv.org/abs/1810.03993, model card format is as follow:
    • Model Details. Basic information about the model.
        – Person or organization developing model
        – Model date
        – Model version
        – Model type
        – Information about training algorithms, parameters, fairness constraints or other applied approaches, and features
        – Paper or other resource for more information
        – Citation details
        – License
        – Where to send questions or comments about the model
        • Intended Use. Use cases that were envisioned during development.
        – Primary intended uses
        – Primary intended users
        – Out-of-scope use cases
        • Factors. Factors could include demographic or phenotypic
        groups, environmental conditions, technical attributes, or
        others listed in Section 4.3.
        – Relevant factors
        – Evaluation factors
        • Metrics. Metrics should be chosen to reflect potential realworld impacts of the model.
        – Model performance measures
        – Decision thresholds
        – Variation approaches
        • Evaluation Data. Details on the dataset(s) used for the
        quantitative analyses in the card.
        – Datasets
        – Motivation
        – Preprocessing
        • Training Data. May not be possible to provide in practice.
        When possible, this section should mirror Evaluation Data.
        If such detail is not possible, minimal allowable information
        should be provided here, such as details of the distribution
        over various factors in the training datasets.
        • Quantitative Analyses
        – Unitary results
        – Intersectional results
        • Ethical Considerations
        • Caveats and Recommendations

    Note: not all information are added, only those which could be extracted from model itself are
    added to model card
    Parameters:
    """

    def __init__(self, **kwargs):
        warnings.warn(
            "Currently only supports for FARMREADER models.", FutureWarning
        )
        self.model_decription = kwargs.get("model_description", None)
        self.model_type = kwargs.get("model_type", None)
        self.model_details = kwargs.get("model_details", None)
        self.intended_use = kwargs.get("intended_use", None)
        self.factors = kwargs.get("factors", None)
        self.metrics = kwargs.get("metrics", None)
        self.evaluation_data = kwargs.get("evaluation_data",None)
        self.training_data = kwargs.get("training_data", None)
        self.quantitative_analyses = kwargs.pop("quantitative_analyses",None)
        self.ethical_considerations = kwargs.pop("ethical_considerations",None)
        self.caveats_and_recommendations = kwargs.pop("caveats_and_recommendations",None)


    def insert_metadata(self):
      """
      this function extracts values for metadata and returns them in dictionary form
      """

      metadata = dict()
      language = self.model_details.get("language", None)
      #not sure which liscence to use
      #license = self.model_details.get("license", "Open Software License 3.0")
      tags = self.model_details.get("tags", None)
      datasets = self.model_details.get("datasets", [])
      metrics = self.model_details.get("metrics", None)
      thumbnail = self.model_details.get("thumbnail", None)


      metadata["language"] = language
      #metadata["license"] = license


      #this is temporary default tags, need to think logic to either automatically generating tags
      if not tags:
        metadata["tags"] = _DEFAULT_TAGS.get(self.model_type, "Not available")
      else:
        metadata["tags"] = tags
      metadata["datasets"] = datasets
      if not metrics:
          metadata["metrics"] = _MODEL_METRIC.get(self.model_type, "Not available")
      else:
        metadata["metrics"] = metrics

      metadata["thumbnail"] = thumbnail

      return metadata

    def model_card_content(self):
      """
        main function to create overall metadata content

      """

      model_card = dict()
      model_card["metadata"] = self.insert_metadata()
      model_card["model_type"] = self.model_type
      model_card["intended_use"] = self.intended_use
      model_card["factors"] = self.factors
      model_card["training_data"] = self.training_data
      model_card["evaluation_data"] = self.evaluation_data
      model_card["quantitative_analyses"] = self.quantitative_analyses
      model_card["ethical_considerations"] = self.ethical_considerations
      model_card["caveats_and_recommendations"] = self.caveats_and_recommendations

      return model_card


    def save_card(self, model_card, card_directory):
      """
        this function takes markdown text and dump it in a file

      """

      with open("README.md", "w") as f:
          f.write(model_card)
      with open(card_directory, "w") as f:
          f.write(model_card)
      return card_directory

    def section_loop(self, keyvalue, model_card):
      for key, value in self.keyvalue.items():
            model_card += f"- {name}: {value}"
            model_card += "\n"
      return model_card


    def generate_model_card(self, card_directory):
      """
            it takes card directory to be saved in and created a markdown structured text
            this is manual markdown creation, should be replaced by some libraries to automatically generate the format
      """

      model_card_content = self.model_card_content()
      metadata = yaml.dump(model_card_content["metadata"], sort_keys=False)
      model_card = f"---\n{metadata}---\n"

      model_card += "\n## Model description\n\n "+self.model_decription +" \n"
      model_card += "\n## Model Type\n\n "+self.model_type + "\n"
      model_card += "\n## Model Details ##\n"
      for name, value in self.model_details.items():
        if name not in metadata:
          model_card += f"- {name}: {value}"
          model_card += "\n"

      model_card += "\n## Training\n"
      if self.training_data is not None:
          model_card += "\nThe following are training details:\n"
          model_card = self.section_loop(self.training_data, model_card)

      else:
          model_card += "\nNo Information\n"
      model_card += "\n## Evaluation\n"
      if self.evaluation_data is not None:
          model_card += "\nThe following are evaluation details:\n"
          model_card = self.section_loop(self.training_data, model_card)
      else:
          model_card += "\nNo Information\n"
      ## these info need to be updated and for now are dummy


      if self.quantitative_analyses is not None:
          model_card += "\n## quantitative_analyses\n\n "+self.quantitative_analyses + "\n"
      else:
           model_card += "\n## quantitative_analyses\n\n not filled yet \n"

      if self.ethical_considerations is not None:
          model_card += "\n## ethical_considerations\n\n "+self.ethical_considerations + "\n"
      else:
           model_card += "\n## ethical_considerations\n\n not filled yet \n"

      if self.caveats_and_recommendations is not None:
          model_card += "\n## caveats_and_recommendations\n\n "+self.caveats_and_recommendations + "\n"
      else:
           model_card += "\n## caveats_and_recommendations\n\n not filled yet \n"
      print("Model Card Generated")
      print(model_card)

      return self.save_card(model_card, card_directory)


