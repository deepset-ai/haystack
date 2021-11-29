
"""
SELECT document.id, document.content, document.vector_id 
FROM document 
JOIN meta_document ON document.id = meta_document.document_id 
WHERE document.index = 'haystack_test' AND 

    meta_document.name = 'year' AND 
    meta_document.value IN ('2020') AND 

    document.id = meta_document.document_id"



SELECT document.id, document.content, document.vector_id 
FROM document 
JOIN meta_document ON document.id = meta_document.document_id 
WHERE document.index = 'haystack_test' AND 

    meta_document.name = 'month' AND 
    meta_document.value IN ('01') AND 

    document.id = meta_document.document_id AND 

    meta_document.name = 'year' AND 
    meta_document.value IN ('2020') AND 

    document.id = meta_document.document_id




SELECT document.id, document.content, document.vector_id 
FROM document 

    JOIN meta_document AS meta_document_1 ON document.id = meta_document_1.document_id 
    JOIN meta_document AS meta_document_2 ON document.id = meta_document_2.document_id 

WHERE document.index = 'haystack_test' AND 

    document.id = meta_document_1.document_id AND 
    meta_document_1.name = 'month' AND 
    meta_document_1.value IN ('01') AND 

    document.id = meta_document_2.document_id AND 
    meta_document_2.name = 'year' AND 
    meta_document_2.value IN ('2020')

"""

print("==============================> ", str(documents_query.statement.compile(compile_kwargs={"literal_binds": True})))
