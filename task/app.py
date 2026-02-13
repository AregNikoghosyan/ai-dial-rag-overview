import os
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr

from task._constants import DIAL_URL, API_KEY


INDEX_NAME = "microwave_faiss_index"


SYSTEM_PROMPT = """You are a professional microwave manual assistant.

You MUST:
- Answer ONLY using provided RAG context.
- Cite the chunk number when referencing information.
- If answer is not in the context, say:
  "I cannot find this information in the microwave manual."
- Reject unrelated questions politely.
"""


USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION:
{query}
"""

class MicrowaveRAG:

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.embeddings = embeddings
        self.llm_client = llm_client
        self.vectorstore = self._setup_vectorstore()


    def _setup_vectorstore(self) -> VectorStore:
        print("🔄 Initializing Microwave Manual RAG System...")

        if os.path.exists(INDEX_NAME):
            print("✅ Loading existing FAISS index...")
            return FAISS.load_local(
                folder_path=INDEX_NAME,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )

        print("🆕 Creating new FAISS index...")
        return self._create_new_index()



    def _create_new_index(self) -> VectorStore:
        print("📖 Loading microwave manual...")

        loader = TextLoader(
            file_path="task/microwave_manual.txt",
            encoding="utf-8"
        )

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."]
        )

        chunks = splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i + 1

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        vectorstore.save_local(INDEX_NAME)

        print("💾 Index created and saved.")
        return vectorstore


    def _is_valid_question(self, query: str) -> bool:
        """
        Basic domain filtering.
        Prevents dinosaur / random questions.
        """

        allowed_keywords = [
            "microwave",
            "oven",
            "cook",
            "clean",
            "install",
            "safety",
            "grill",
            "eco",
            "time",
            "manual"
        ]

        query_lower = query.lower()

        return any(keyword in query_lower for keyword in allowed_keywords)

    def retrieve_context(self, query: str, k: int = 4, score_threshold: float = 0.3) -> str:

        print("\n" + "=" * 100)
        print("🔍 STEP 1: RETRIEVAL")
        print("-" * 100)
        print(f"Query: {query}")

        results = self.vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k=k
    )

        context_parts = []

        for doc, score in results:
            if score < score_threshold:
                continue

            chunk_id = doc.metadata.get("chunk_id", "N/A")

            print(f"\nChunk #{chunk_id} | Score: {score}")
            print(doc.page_content)

            formatted_chunk = f"[Chunk {chunk_id}]\n{doc.page_content}"
            context_parts.append(formatted_chunk)

        if not context_parts:
            print("⚠ No relevant context found.")
            return ""

        print("=" * 100)
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        print("\n🔗 STEP 2: AUGMENTATION")
        print("-" * 100)

        if not context:
            return ""

        prompt = USER_PROMPT.format(
            context=context,
            query=query
        )

        print(prompt)
        print("=" * 100)

        return prompt


    def generate_answer(self, augmented_prompt: str) -> str:

        print("\n🤖 STEP 3: GENERATION")
        print("-" * 100)

        if not augmented_prompt:
            return "I cannot find this information in the microwave manual."

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]

        response = self.llm_client.invoke(messages)

        print(response.content)
        print("=" * 100)

        return response.content


def main(rag: MicrowaveRAG):
    print("\n🎯 Microwave RAG Assistant Ready (type 'exit' to quit)\n")

    while True:
        user_question = input("> ").strip()

        if user_question.lower() == "exit":
            break

        if not rag._is_valid_question(user_question):
            print("This question is outside the microwave manual domain.")
            continue

        context = rag.retrieve_context(user_question)
        augmented_prompt = rag.augment_prompt(user_question, context)
        rag.generate_answer(augmented_prompt)


main(
    MicrowaveRAG(
        embeddings=AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small-1",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY)
        ),
        llm_client=AzureChatOpenAI(
            temperature=0.0,
            azure_deployment="gpt-4o",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            api_version=""
        )
    )
)
