"""
Agent 3: Translation & Processing Agent
Handles detection and translation of non-English documents.

Reads from:
- state["documents"]

Writes to (in-place modification):
- doc["content"] (translated to English)
- doc["title"] (translated to English)
- doc["language"] (set to "en")
- doc["translated"] (set to True)
- doc["metadata"]["original_language"]
- doc["metadata"]["original_title"]
- doc["metadata"]["translation_model"]
"""

from langchain_core.prompts import ChatPromptTemplate

from state import ActiveWarningsState
from llm import llm


# A set of common English language identifiers from Seerist and ReliefWeb
ENGLISH_LANG_CODES = {"en", "eng", "english"}


# A simple, direct prompt for translation.
# The design doc [cite: 208] suggests GPT-4 or specialized models.
TRANSLATION_PROMPT_TEMPLATE = """
You are a professional, high-quality translator.
Translate the following text into English.
Return *only* the translated text, with no preamble, apologies, or explanations.

Text to translate:
---
{text_to_translate}
---
"""


def run_translation_agent(state: ActiveWarningsState) -> ActiveWarningsState:
    """LangGraph node function to translate non-English documents.

    This function:
    1. Iterates through all documents in state["documents"].
    2. Checks the "language" field.
    3. If not English, it translates the "title" and "content" to English.
    4. It updates the document in-place, setting "translated" to True
       and populating "metadata" with translation notes. [cite: 45, 46]
    """
    print(f"--- (3) Running Translation & Processing Agent for {state['country']} ---")

    if state.get("documents") is None or not state["documents"]:
        print("   > No documents found to translate. Skipping.")
        state["current_step"] = "TranslationComplete"
        return state

    # Initialize warnings list if needed
    if state.get("warnings") is None:
        state["warnings"] = []

    try:
        # Set up the translation chain (assumes 'llm' is defined)
        translation_prompt = ChatPromptTemplate.from_template(TRANSLATION_PROMPT_TEMPLATE)
        translation_chain = translation_prompt | llm

        documents = state["documents"]
        translated_count = 0

        for doc in documents:
            original_language = doc.get("language", "en").lower()

            # Check if translation is needed
            if original_language not in ENGLISH_LANG_CODES and doc.get("content"):
                if translated_count == 0:
                    print("   > Detected non-English content. Starting translation...")

                print(f"      ... Translating doc {doc['doc_id']} from '{original_language}'")
                translated_count += 1

                # 1. Store original data for metadata
                original_title = doc.get("title", "")
                original_content = doc.get("content", "")

                # 2. Translate content
                content_response = translation_chain.invoke({"text_to_translate": original_content})
                translated_content = content_response.content

                # 3. Translate title
                title_response = translation_chain.invoke({"text_to_translate": original_title})
                translated_title = title_response.content

                # 4. Update the document in-place
                doc["content"] = translated_content
                doc["title"] = translated_title
                doc["language"] = "en"
                doc["translated"] = True

                # 5. Maintain translation metadata
                if doc.get("metadata") is None:
                    doc["metadata"] = {}

                doc["metadata"]["original_language"] = original_language
                doc["metadata"]["original_title"] = original_title
                doc["metadata"]["translation_model"] = getattr(llm, "model", "unknown")

                # The design doc mentions confidence, but standard LLMs don't provide this.
                # A specialized API would. We'll leave it as None per the TypedDict.
                doc["translation_confidence"] = None

        print(f"   > Translation complete. {translated_count} documents were translated.")
        state["current_step"] = "TranslationComplete"

    except Exception as e:  # noqa: BLE001
        print(f"   ! ERROR in Translation Agent: {e}")
        state["warnings"].append(f"TranslationAgentError: {str(e)}")

    return state
