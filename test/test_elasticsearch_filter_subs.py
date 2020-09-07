def test_filter_subs():
    import json
    import string
    custom_query = """
                 {
                       "size":10,
                       "query":{
                          "bool":{
                             "should":[
                                {
                                   "multi_match":{
                                      "query":"${question}",
                                      "type":"most_fields",
                                      "fields":[
                                         "text"
                                      ]
                                   }
                                }
                             ],
                             "filter":[
                                {
                                   "terms":{
                                      "source_name": ${source_names}
                                   }
                                }
                             ]
                          }
                       }
                    }
                """

    template = string.Template(custom_query)
    # replace all "${question}" placeholder(s) with query
    substitutions = {"question": "what is X"}
    # For each filter we got passed, we'll try to find & replace the corresponding placeholder in the template
    # Example: filters={"years":[2018]} => replaces {$years} in custom_query with '[2018]'
    filters = {"source_names": ["test"]}
    if filters:
        for key, values in filters.items():
            values_str = json.dumps(values)
            substitutions[key] = values_str
    custom_query_json = template.substitute(**substitutions)
    body = json.loads(custom_query_json)
    # add top_k
    body["size"] = str(10)
    assert body == {'query': {'bool': {'filter': [{'terms': {'source_name': ['test']}}],
                    'should': [{'multi_match': {'fields': ['text'],
                                                'query': 'what is X',
                                                'type': 'most_fields'}}]}}, 'size': '10'}