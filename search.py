import streamlit as st
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import BM25Retriever
import os
from haystack.nodes import TextConverter
from haystack.nodes import PreProcessor
# from haystack.document_stores import SQLDocumentStore
from haystack.pipelines.standard_pipelines import TextIndexingPipeline

# File converter 
converter = TextConverter(remove_numeric_tables=True, valid_languages=["en"])

document_store = InMemoryDocumentStore(use_bm25=True)

# Preprocessor for text cleaning
preprocessor = PreProcessor(clean_empty_lines=True,
                            clean_whitespace=True,
                            clean_header_footer=False,
                            split_by="word",
                            split_length=1000,
                            split_respect_sentence_boundary=True
                           )

doc_dir = r'path to your docs'
# doc = preprocessor.process([doc_dir])
files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]

indexing_pipeline = TextIndexingPipeline(preprocessor)

indexing_pipeline = TextIndexingPipeline(document_store)

indexing_pipeline.run_batch(file_paths=files_to_index)

retriever = BM25Retriever(document_store=document_store)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2",use_gpu=True)

pipe = ExtractiveQAPipeline(reader, retriever) # Prediction pipeline 

st.title("Question Answering system")
question = st.text_area("Enter your question:") # taking query
if st.button("Answer"):
    params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}}
    prediction = pipe.run(query=question, params=params)
    
    # show results
    for ans in prediction['answers']:
        st.write(ans.answer) # main answer
        st.write(ans.context) # context
        st.write('---')