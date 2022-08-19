#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:05:13 2022

@author: adithya

This file contains an extensible template for the model card. 
Class Object is created with vars such that final dictionary can be converted to YAML file
Once all fields are populated, call generate model card to generate description and create the final README.md
"""

import warnings
import yaml

class ModelCard():

    def __init__(self):
        warnings.warn(
            "Currently only supports for FARMREADER models.", FutureWarning
        )
        self.model_card = dict()
        self.model_card['language'] = 'en'
        self.model_card['datasets'] = []
        self.model_card['license'] = ''
        
        self.model_card['model_index'] = dict()
        self.model_card['model_index']['name'] = '' 
        
        self.model_card['model_index']['results'] = dict()
        self.model_card['model_index']['results']['task'] = dict()
        self.model_card['model_index']['results']['task']['type'] = '' 
        if self.model_card['model_index']['results']['task']['type'].lower() == 'question_answering':
            self.model_card['model_index']['results']['task']['name'] = 'Question Answering'
        else:
            self.model_card['model_index']['results']['task']['name'] = ''
        
        self.model_card['model_index']['results']['dataset'] = dict()
        
        self.model_card['model_index']['results']['dataset']['name'] = ''
        self.model_card['model_index']['results']['dataset']['type'] = ''
        self.model_card['model_index']['results']['dataset']['config'] = ''
        self.model_card['model_index']['results']['dataset']['split'] = ''

        self.model_card['model_index']['results']['metrics'] = []
        metrics_dict = dict() 
        metrics_dict['name']= 'Exact Match'
        metrics_dict['type']= 'exact_match'
        metrics_dict['value']= 0
        metrics_dict['verified']= False
        self.model_card['model_index']['results']['metrics'].append(metrics_dict)
        
        metrics_dict = dict() 
        metrics_dict['name']= 'f1'
        metrics_dict['type']= 'f1'
        metrics_dict['value']= 0.0
        metrics_dict['verified']= False
        self.model_card['model_index']['results']['metrics'].append(metrics_dict)
        
        metrics_dict = dict() 
        metrics_dict['name']= 'total'
        metrics_dict['name']= 'total'
        metrics_dict['value']= 0
        metrics_dict['verified']= False
        self.model_card['model_index']['results']['metrics'].append(metrics_dict)
        
        
        self.overview = dict()
        self.overview['Language_model']  = '' 
        self.overview['Language'] ='English'
        self.overview['Downstream_task'] =''  
        self.overview['Training_data']  =''
        self.overview['Eval_data']  =''
        self.overview['Code']=''
        self.overview['Infrastructure'] =''
        
        self.hyperparams = dict()
        self.hyperparams['batch_size ']  = 0
        self.hyperparams['n_epochs '] = 0
        self.hyperparams['base_LM_model']  = ''
        self.hyperparams['max_seq_len']  = 0
        self.hyperparams['learning_rate']  = 0 
        self.hyperparams['lr_schedule'] = ''
        self.hyperparams['warmup_proportion'] = ''
        self.hyperparams['doc_stride'] = 0


    def generate_description(self):    
        self.description = 'This is the ['+str(self.overview['Language_model']) + '] model, fine-tuned using the [' + str(self.datasets) + "] dataset. It has been trained on [" + str(self.model_card['model_index']['results']['task']['type']) + "] pairs, including unanswerable questions, for the task of ["+str(self.model_card['model_index']['results']['task']['name']) +"]."

    def save_card(self, model_card):

      CARD_DIRECTORY = "README.md"

      with open("README.md", "w") as f:
        f.write(model_card)
      return CARD_DIRECTORY

    def generate_model_card(self):

      model_card_content = yaml.dump(self.model_card, sort_keys = False)
      
      self.generate_description()
      model_card_content += "\n## Model description\n\n "+self.decription +" \n"
      
      model_card_content += "\n## Overview\n\n "+ yaml.dump(self.overview, sort_keys = False)+ "\n"
      
      model_card_content += "\n## Hyper parameters\n\n "+ yaml.dump(self.hyperparams, sort_keys = False)+ "\n"

      print(model_card_content)

      return self.save_card(model_card_content)
