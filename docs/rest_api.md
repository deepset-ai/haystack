<a name="application"></a>
# application

<a name="__init__"></a>
# \_\_init\_\_

<a name="config"></a>
# config

<a name="controller"></a>
# controller

<a name="controller.search"></a>
# controller.search

<a name="controller.feedback"></a>
# controller.feedback

<a name="controller.feedback.export_doc_qa_feedback"></a>
#### export\_doc\_qa\_feedback

```python
@router.get("/export-doc-qa-feedback")
export_doc_qa_feedback(context_size: int = 2_000)
```

SQuAD format JSON export for question/answer pairs that were marked as "relevant".

The context_size param can be used to limit response size for large documents.

<a name="controller.feedback.export_faq_feedback"></a>
#### export\_faq\_feedback

```python
@router.get("/export-faq-qa-feedback")
export_faq_feedback()
```

Export feedback for faq-qa in JSON format.

<a name="controller.router"></a>
# controller.router

<a name="controller.utils"></a>
# controller.utils

<a name="controller.file_upload"></a>
# controller.file\_upload

<a name="controller.errors"></a>
# controller.errors

<a name="controller.errors.http_error"></a>
# controller.errors.http\_error

