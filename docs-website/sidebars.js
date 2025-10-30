// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

export default {
  docs: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Overview',
      items: [
        'overview/installation',
        'overview/get-started',
        'overview/faq',
        'overview/telemetry',
        'overview/breaking-change-policy',
        'overview/migration',
      ],
    },
    {
      type: 'category',
      label: 'Haystack Concepts',
      link: {
        type: 'doc',
        id: 'concepts/concepts-overview'
      },
      items: [
        {
          type: 'category',
          label: 'Components',
          link: {
            type: 'doc',
            id: 'concepts/components'
          },
          items: [
            'concepts/components/custom-components',
            'concepts/components/supercomponents',
          ],
        },
        'concepts/components-overview',
        {
          type: 'category',
          label: 'Data Classes',
          link: {
            type: 'doc',
            id: 'concepts/data-classes'
          },
          items: [
            'concepts/data-classes/chatmessage',
          ],
        },
        {
          type: 'category',
          label: 'Document Store',
          link: {
            type: 'doc',
            id: 'concepts/document-store'
          },
          items: [
            'concepts/document-store/choosing-a-document-store',
            'concepts/document-store/creating-custom-document-stores',
          ],
        },
        {
          type: 'category',
          label: 'Pipelines',
          link: {
            type: 'doc',
            id: 'concepts/pipelines'
          },
          items: [
            'concepts/pipelines/creating-pipelines',
            'concepts/pipelines/serialization',
            'concepts/pipelines/visualizing-pipelines',
            'concepts/pipelines/debugging-pipelines',
            'concepts/pipelines/pipeline-breakpoints',
            'concepts/pipelines/pipeline-templates',
            'concepts/pipelines/asyncpipeline',
          ],
        },
        {
          type: 'category',
          label: 'Agents',
          link: {
            type: 'doc',
            id: 'concepts/agents'
          },
          items: [
            'concepts/agents/state',
          ],
        },
        'concepts/integrations',
        'concepts/jinja-templates',
        'concepts/metadata-filtering',
        'concepts/device-management',
        'concepts/secret-management',
        'concepts/experimental-package',
      ],
    },
    {
      type: 'category',
      label: 'Document Stores',
      items: [
        'document-stores/inmemorydocumentstore',
        'document-stores/astradocumentstore',
        'document-stores/azureaisearchdocumentstore',
        'document-stores/chromadocumentstore',
        {
          type: 'link',
          label: 'CouchbaseDocumentStore',
          href: 'https://haystack.deepset.ai/integrations/couchbase-document-store',
        },
        'document-stores/elasticsearch-document-store',
        {
          type: 'link',
          label: 'LanceDBDocumentStore',
          href: 'https://haystack.deepset.ai/integrations/lancedb/',
        },
        {
          type: 'link',
          label: 'MarqoDocumentStore',
          href: 'https://haystack.deepset.ai/integrations/marqo-document-store/',
        },
        {
          type: 'link',
          label: 'MilvusDocumentStore',
          href: 'https://haystack.deepset.ai/integrations/milvus-document-store',
        },
        'document-stores/mongodbatlasdocumentstore',
        {
          type: 'link',
          label: 'Neo4jDocumentStore',
          href: 'https://haystack.deepset.ai/integrations/neo4j-document-store',
        },
        'document-stores/opensearch-document-store',
        'document-stores/pgvectordocumentstore',
        'document-stores/pinecone-document-store',
        'document-stores/qdrant-document-store',
        'document-stores/weaviatedocumentstore',
      ],
    },
    {
      type: 'category',
      label: 'Pipeline Components',
      items: [
        {
          type: 'category',
          label: 'Agents',
          items: [
            'pipeline-components/agents-1/agent',
          ],
        },
        {
          type: 'category',
          label: 'Audio',
          link: {
            type: 'doc',
            id: 'pipeline-components/audio'
          },
          items: [
            'pipeline-components/audio/external-integrations-audio',
            'pipeline-components/audio/localwhispertranscriber',
            'pipeline-components/audio/remotewhispertranscriber',
          ],
        },
        {
          type: 'category',
          label: 'Builders',
          link: {
            type: 'doc',
            id: 'pipeline-components/builders'
          },
          items: [
            'pipeline-components/builders/answerbuilder',
            'pipeline-components/builders/chatpromptbuilder',
            'pipeline-components/builders/promptbuilder',
          ],
        },
        {
          type: 'category',
          label: 'Caching',
          items: [
            'pipeline-components/caching/cachechecker',
          ],
        },
        {
          type: 'category',
          label: 'Classifiers',
          link: {
            type: 'doc',
            id: 'pipeline-components/classifiers'
          },
          items: [
            'pipeline-components/classifiers/documentlanguageclassifier',
            'pipeline-components/classifiers/transformerszeroshotdocumentclassifier',
          ],
        },
        {
          type: 'category',
          label: 'Connectors',
          link: {
            type: 'doc',
            id: 'pipeline-components/connectors'
          },
          items: [
            'pipeline-components/connectors/external-integrations-connectors',
            'pipeline-components/connectors/githubfileeditor',
            'pipeline-components/connectors/githubissuecommenter',
            'pipeline-components/connectors/githubissueviewer',
            'pipeline-components/connectors/githubprcreator',
            'pipeline-components/connectors/githubrepoforker',
            'pipeline-components/connectors/githubrepoviewer',
            'pipeline-components/connectors/jinareaderconnector',
            'pipeline-components/connectors/langfuseconnector',
            'pipeline-components/connectors/openapiconnector',
            'pipeline-components/connectors/openapiserviceconnector',
            'pipeline-components/connectors/weaveconnector',
          ],
        },
        {
          type: 'category',
          label: 'Converters',
          link: {
            type: 'doc',
            id: 'pipeline-components/converters'
          },
          items: [
            'pipeline-components/converters/azureocrdocumentconverter',
            'pipeline-components/converters/csvtodocument',
            'pipeline-components/converters/documenttoimagecontent',
            'pipeline-components/converters/docxtodocument',
            'pipeline-components/converters/external-integrations-converters',
            'pipeline-components/converters/htmltodocument',
            'pipeline-components/converters/imagefiletodocument',
            'pipeline-components/converters/imagefiletoimagecontent',
            'pipeline-components/converters/jsonconverter',
            'pipeline-components/converters/markdowntodocument',
            'pipeline-components/converters/mistralocrdocumentconverter',
            'pipeline-components/converters/msgtodocument',
            'pipeline-components/converters/multifileconverter',
            'pipeline-components/converters/openapiservicetofunctions',
            'pipeline-components/converters/outputadapter',
            'pipeline-components/converters/pdfminertodocument',
            'pipeline-components/converters/pdftoimagecontent',
            'pipeline-components/converters/pptxtodocument',
            'pipeline-components/converters/pypdftodocument',
            'pipeline-components/converters/textfiletodocument',
            'pipeline-components/converters/tikadocumentconverter',
            'pipeline-components/converters/unstructuredfileconverter',
            'pipeline-components/converters/xlsxtodocument',
          ],
        },
        {
          type: 'category',
          label: 'Downloaders',
          items: [
            'pipeline-components/downloaders/s3downloader',
          ],
        },
        {
          type: 'category',
          label: 'Embedders',
          link: {
            type: 'doc',
            id: 'pipeline-components/embedders'
          },
          items: [
            'pipeline-components/embedders/amazonbedrockdocumentembedder',
            'pipeline-components/embedders/amazonbedrockdocumentimageembedder',
            'pipeline-components/embedders/amazonbedrocktextembedder',
            'pipeline-components/embedders/azureopenaidocumentembedder',
            'pipeline-components/embedders/azureopenaitextembedder',
            'pipeline-components/embedders/choosing-the-right-embedder',
            'pipeline-components/embedders/coheredocumentembedder',
            'pipeline-components/embedders/coheredocumentimageembedder',
            'pipeline-components/embedders/coheretextembedder',
            'pipeline-components/embedders/external-integrations-embedders',
            'pipeline-components/embedders/fastembeddocumentembedder',
            'pipeline-components/embedders/fastembedsparsedocumentembedder',
            'pipeline-components/embedders/fastembedsparsetextembedder',
            'pipeline-components/embedders/fastembedtextembedder',
            'pipeline-components/embedders/googlegenaidocumentembedder',
            'pipeline-components/embedders/googlegenaitextembedder',
            'pipeline-components/embedders/huggingfaceapidocumentembedder',
            'pipeline-components/embedders/huggingfaceapitextembedder',
            'pipeline-components/embedders/jinadocumentembedder',
            'pipeline-components/embedders/jinadocumentimageembedder',
            'pipeline-components/embedders/jinatextembedder',
            'pipeline-components/embedders/mistraldocumentembedder',
            'pipeline-components/embedders/mistraltextembedder',
            'pipeline-components/embedders/nvidiadocumentembedder',
            'pipeline-components/embedders/nvidiatextembedder',
            'pipeline-components/embedders/ollamadocumentembedder',
            'pipeline-components/embedders/ollamatextembedder',
            'pipeline-components/embedders/openaidocumentembedder',
            'pipeline-components/embedders/openaitextembedder',
            'pipeline-components/embedders/optimumdocumentembedder',
            'pipeline-components/embedders/optimumtextembedder',
            'pipeline-components/embedders/sentencetransformersdocumentembedder',
            'pipeline-components/embedders/sentencetransformersdocumentimageembedder',
            'pipeline-components/embedders/sentencetransformerssparsedocumentembedder',
            'pipeline-components/embedders/sentencetransformerssparsetextembedder',
            'pipeline-components/embedders/sentencetransformerstextembedder',
            'pipeline-components/embedders/stackitdocumentembedder',
            'pipeline-components/embedders/stackittextembedder',
            'pipeline-components/embedders/vertexaidocumentembedder',
            'pipeline-components/embedders/vertexaitextembedder',
            'pipeline-components/embedders/watsonxdocumentembedder',
            'pipeline-components/embedders/watsonxtextembedder',
          ],
        },
        {
          type: 'category',
          label: 'Evaluators',
          link: {
            type: 'doc',
            id: 'pipeline-components/evaluators'
          },
          items: [
            'pipeline-components/evaluators/answerexactmatchevaluator',
            'pipeline-components/evaluators/contextrelevanceevaluator',
            'pipeline-components/evaluators/deepevalevaluator',
            'pipeline-components/evaluators/documentmapevaluator',
            'pipeline-components/evaluators/documentmrrevaluator',
            'pipeline-components/evaluators/documentndcgevaluator',
            'pipeline-components/evaluators/documentrecallevaluator',
            'pipeline-components/evaluators/external-integrations-evaluators',
            'pipeline-components/evaluators/faithfulnessevaluator',
            'pipeline-components/evaluators/llmevaluator',
            'pipeline-components/evaluators/ragasevaluator',
            'pipeline-components/evaluators/sasevaluator',
          ],
        },
        {
          type: 'category',
          label: 'Extractors',
          link: {
            type: 'doc',
            id: 'pipeline-components/extractors'
          },
          items: [
            'pipeline-components/extractors/llmdocumentcontentextractor',
            'pipeline-components/extractors/llmmetadataextractor',
            'pipeline-components/extractors/namedentityextractor',
          ],
        },
        {
          type: 'category',
          label: 'Fetchers',
          link: {
            type: 'doc',
            id: 'pipeline-components/fetchers'
          },
          items: [
            'pipeline-components/fetchers/external-integrations-fetchers',
            'pipeline-components/fetchers/linkcontentfetcher',
          ],
        },
        {
          type: 'category',
          label: 'Generators',
          link: {
            type: 'doc',
            id: 'pipeline-components/generators'
          },
          items: [
            'pipeline-components/generators/amazonbedrockchatgenerator',
            'pipeline-components/generators/amazonbedrockgenerator',
            'pipeline-components/generators/anthropicchatgenerator',
            'pipeline-components/generators/anthropicgenerator',
            'pipeline-components/generators/anthropicvertexchatgenerator',
            'pipeline-components/generators/azureopenaichatgenerator',
            'pipeline-components/generators/azureopenaigenerator',
            'pipeline-components/generators/coherechatgenerator',
            'pipeline-components/generators/coheregenerator',
            'pipeline-components/generators/dalleimagegenerator',
            'pipeline-components/generators/external-integrations-generators',
            'pipeline-components/generators/fallbackchatgenerator',
            'pipeline-components/generators/googleaigeminichatgenerator',
            'pipeline-components/generators/googleaigeminigenerator',
            'pipeline-components/generators/googlegenaichatgenerator',
            {
              type: 'category',
              label: 'Guides to Generators',
              items: [
                'pipeline-components/generators/guides-to-generators/choosing-the-right-generator',
                'pipeline-components/generators/guides-to-generators/function-calling',
                'pipeline-components/generators/guides-to-generators/generators-vs-chat-generators',
              ],
            },
            'pipeline-components/generators/huggingfaceapichatgenerator',
            'pipeline-components/generators/huggingfaceapigenerator',
            'pipeline-components/generators/huggingfacelocalchatgenerator',
            'pipeline-components/generators/huggingfacelocalgenerator',
            'pipeline-components/generators/llamacppchatgenerator',
            'pipeline-components/generators/llamacppgenerator',
            'pipeline-components/generators/llamastackchatgenerator',
            'pipeline-components/generators/metallamachatgenerator',
            'pipeline-components/generators/mistralchatgenerator',
            'pipeline-components/generators/nvidiachatgenerator',
            'pipeline-components/generators/nvidiagenerator',
            'pipeline-components/generators/ollamachatgenerator',
            'pipeline-components/generators/ollamagenerator',
            'pipeline-components/generators/openaichatgenerator',
            'pipeline-components/generators/openaigenerator',
            'pipeline-components/generators/openrouterchatgenerator',
            'pipeline-components/generators/sagemakergenerator',
            'pipeline-components/generators/stackitchatgenerator',
            'pipeline-components/generators/togetheraichatgenerator',
            'pipeline-components/generators/togetheraigenerator',
            'pipeline-components/generators/vertexaicodegenerator',
            'pipeline-components/generators/vertexaigeminichatgenerator',
            'pipeline-components/generators/vertexaigeminigenerator',
            'pipeline-components/generators/vertexaiimagecaptioner',
            'pipeline-components/generators/vertexaiimagegenerator',
            'pipeline-components/generators/vertexaiimageqa',
            'pipeline-components/generators/vertexaitextgenerator',
            'pipeline-components/generators/watsonxchatgenerator',
            'pipeline-components/generators/watsonxgenerator',
          ],
        },
        {
          type: 'category',
          label: 'Joiners',
          link: {
            type: 'doc',
            id: 'pipeline-components/joiners'
          },
          items: [
            'pipeline-components/joiners/answerjoiner',
            'pipeline-components/joiners/branchjoiner',
            'pipeline-components/joiners/documentjoiner',
            'pipeline-components/joiners/listjoiner',
            'pipeline-components/joiners/stringjoiner',
          ],
        },
        {
          type: 'category',
          label: 'Preprocessors',
          link: {
            type: 'doc',
            id: 'pipeline-components/preprocessors'
          },
          items: [
            'pipeline-components/preprocessors/chinesedocumentsplitter',
            'pipeline-components/preprocessors/csvdocumentcleaner',
            'pipeline-components/preprocessors/csvdocumentsplitter',
            'pipeline-components/preprocessors/documentcleaner',
            'pipeline-components/preprocessors/documentpreprocessor',
            'pipeline-components/preprocessors/documentsplitter',
            'pipeline-components/preprocessors/hierarchicaldocumentsplitter',
            'pipeline-components/preprocessors/recursivesplitter',
            'pipeline-components/preprocessors/textcleaner',
          ],
        },
        {
          type: 'category',
          label: 'Rankers',
          link: {
            type: 'doc',
            id: 'pipeline-components/rankers'
          },
          items: [
            'pipeline-components/rankers/amazonbedrockranker',
            'pipeline-components/rankers/choosing-the-right-ranker',
            'pipeline-components/rankers/cohereranker',
            'pipeline-components/rankers/external-integrations-rankers',
            'pipeline-components/rankers/fastembedranker',
            'pipeline-components/rankers/huggingfaceteiranker',
            'pipeline-components/rankers/jinaranker',
            'pipeline-components/rankers/lostinthemiddleranker',
            'pipeline-components/rankers/metafieldgroupingranker',
            'pipeline-components/rankers/metafieldranker',
            'pipeline-components/rankers/nvidiaranker',
            'pipeline-components/rankers/sentencetransformersdiversityranker',
            'pipeline-components/rankers/sentencetransformerssimilarityranker',
            'pipeline-components/rankers/transformerssimilarityranker',
          ],
        },
        {
          type: 'category',
          label: 'Readers',
          link: {
            type: 'doc',
            id: 'pipeline-components/readers'
          },
          items: [
            'pipeline-components/readers/extractivereader',
          ],
        },
        {
          type: 'category',
          label: 'Retrievers',
          link: {
            type: 'doc',
            id: 'pipeline-components/retrievers'
          },
          items: [
            'pipeline-components/retrievers/astraretriever',
            'pipeline-components/retrievers/automergingretriever',
            'pipeline-components/retrievers/azureaisearchbm25retriever',
            'pipeline-components/retrievers/azureaisearchembeddingretriever',
            'pipeline-components/retrievers/azureaisearchhybridretriever',
            'pipeline-components/retrievers/chromaembeddingretriever',
            'pipeline-components/retrievers/chromaqueryretriever',
            'pipeline-components/retrievers/elasticsearchbm25retriever',
            'pipeline-components/retrievers/elasticsearchembeddingretriever',
            'pipeline-components/retrievers/filterretriever',
            'pipeline-components/retrievers/inmemorybm25retriever',
            'pipeline-components/retrievers/inmemoryembeddingretriever',
            'pipeline-components/retrievers/mongodbatlasembeddingretriever',
            'pipeline-components/retrievers/mongodbatlasfulltextretriever',
            'pipeline-components/retrievers/opensearchbm25retriever',
            'pipeline-components/retrievers/opensearchembeddingretriever',
            'pipeline-components/retrievers/opensearchhybridretriever',
            'pipeline-components/retrievers/pgvectorembeddingretriever',
            'pipeline-components/retrievers/pgvectorkeywordretriever',
            'pipeline-components/retrievers/pineconedenseretriever',
            'pipeline-components/retrievers/qdrantembeddingretriever',
            'pipeline-components/retrievers/qdranthybridretriever',
            'pipeline-components/retrievers/qdrantsparseembeddingretriever',
            'pipeline-components/retrievers/sentencewindowretrieval',
            'pipeline-components/retrievers/snowflaketableretriever',
            'pipeline-components/retrievers/weaviatebm25retriever',
            'pipeline-components/retrievers/weaviateembeddingretriever',
            'pipeline-components/retrievers/weaviatehybridretriever',
          ],
        },
        {
          type: 'category',
          label: 'Routers',
          link: {
            type: 'doc',
            id: 'pipeline-components/routers'
          },
          items: [
            'pipeline-components/routers/conditionalrouter',
            'pipeline-components/routers/documentlengthrouter',
            'pipeline-components/routers/documenttyperouter',
            'pipeline-components/routers/filetyperouter',
            'pipeline-components/routers/llmmessagesrouter',
            'pipeline-components/routers/metadatarouter',
            'pipeline-components/routers/textlanguagerouter',
            'pipeline-components/routers/transformerstextrouter',
            'pipeline-components/routers/transformerszeroshottextrouter',
          ],
        },
        {
          type: 'category',
          label: 'Samplers',
          items: [
            'pipeline-components/samplers/toppsampler',
          ],
        },
        {
          type: 'category',
          label: 'Tools',
          items: [
            'pipeline-components/tools/toolinvoker',
          ],
        },
        {
          type: 'category',
          label: 'Validators',
          items: [
            'pipeline-components/validators/jsonschemavalidator',
          ],
        },
        {
          type: 'category',
          label: 'Websearch',
          link: {
            type: 'doc',
            id: 'pipeline-components/websearch'
          },
          items: [
            'pipeline-components/websearch/external-integrations-websearch',
            'pipeline-components/websearch/searchapiwebsearch',
            'pipeline-components/websearch/serperdevwebsearch',
          ],
        },
        {
          type: 'category',
          label: 'Writers',
          items: [
            'pipeline-components/writers/documentwriter',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Tools',
      items: [
        'tools/tool',
        'tools/componenttool',
        'tools/pipelinetool',
        'tools/toolset',
        'tools/mcptool',
        'tools/mcptoolset',
        {
          type: 'category',
          label: 'Ready-made Tools',
          items: [
            'tools/ready-made-tools/githubfileeditortool',
            'tools/ready-made-tools/githubissuecommentertool',
            'tools/ready-made-tools/githubissueviewertool',
            'tools/ready-made-tools/githubprcreatortool',
            'tools/ready-made-tools/githubrepoviewertool',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Optimization',
      items: [
        {
          type: 'category',
          label: 'Evaluation',
          link: {
            type: 'doc',
            id: 'optimization/evaluation'
          },
          items: [
            'optimization/evaluation/model-based-evaluation',
            'optimization/evaluation/statistical-evaluation',
          ],
        },
        {
          type: 'category',
          label: 'Advanced RAG Techniques',
          link: {
            type: 'doc',
            id: 'optimization/advanced-rag-techniques'
          },
          items: [
            'optimization/advanced-rag-techniques/hypothetical-document-embeddings-hyde',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Development',
      items: [
        'development/logging',
        'development/tracing',
        'development/external-integrations-development',
        'development/enabling-gpu-acceleration',
        'development/hayhooks',
        {
          type: 'category',
          label: 'Deployment',
          link: {
            type: 'doc',
            id: 'development/deployment'
          },
          items: [
            'development/deployment/docker',
            'development/deployment/kubernetes',
            'development/deployment/openshift',
          ],
        },
      ],
    },
  ],
};
