import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline

def main():

    hbert_path = "heBERT"
    hebrew_gemma_path = "Hebrew-Gemma-11B-V2"

    print("Loading H-BERT embeddings model...")
    hbert_tokenizer = AutoTokenizer.from_pretrained(hbert_path)
    hbert_model = AutoModel.from_pretrained(hbert_path)

    embeddings = HuggingFaceEmbeddings(model_name=hbert_path)

    documents = [
        "זהו מסמך ראשון לדוגמא. הטקסט בעברית ומכיל מידע כלשהו.",
        "זהו המסמך השני לדוגמא, המכיל מידע על ישראל ועל הכלכלה שלה.",
        "זהו מסמך שלישי על ספורט וכדורגל בישראל."
    ]

    print("Creating FAISS index...")
    faiss_index = FAISS.from_texts(texts=documents, embedding=embeddings)

    print("Loading Hebrew-Gemma LLM...")
    gemma_tokenizer = AutoTokenizer.from_pretrained(hebrew_gemma_path)
    gemma_model = AutoModelForCausalLM.from_pretrained(hebrew_gemma_path,
        device_map="auto",
        low_cpu_mem_usage=True)

    generation_pipeline = pipeline(
        "text-generation",
        model=gemma_model,
        tokenizer=gemma_tokenizer,
    )
    
    llm = HuggingFacePipeline(pipeline=generation_pipeline)
    
    retriever = faiss_index.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    query = "מהו המידע החשוב ביותר על ישראל במסמכים?"
    print(f"\nUser Query: {query}")
    
    result = qa_chain(query)
    answer = result["result"]
    source_docs = result["source_documents"]
    
    print("\n=== Generated Answer ===")
    print(answer)
    
    print("\n=== Source Documents ===")
    for doc in source_docs:
        print("-" * 50)
        print(doc.page_content)

if __name__ == "__main__":
    main()