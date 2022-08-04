from haystack import Pipeline
import tempfile

tmp = tempfile.NamedTemporaryFile()


pipe_yaml="""version: ignore

components:
  - name: MyReader
    type: FARMReader
    params:
      model_name_or_path: deepset/tinyroberta-squad2
      model_version: null
  - name: myopen
    type: OpenAIAnswerGenerator
    params:
      api_key: dfdsfgfsdfsdfsdffdsfdsfsd
      examples: [!!float 123, !!float 123]


pipelines:
  - name: example_pipeline
    nodes:
      - name: MyReader
        inputs: [Query]"""

with open(tmp.name, 'w') as f:
    f.write(pipe_yaml)         

pipe=Pipeline.load_from_yaml(tmp.name)

print(pipe.to_code())

# bisogna modificare la generazione