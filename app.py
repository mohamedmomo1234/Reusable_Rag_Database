import uuid
import streamlit as st

from pathlib import Path

from ingest import create_vector_database
from config import CHROMA_DIR

from config import (
    ADMIN_PASSWORD,
    APP_DESCRIPTION,
    APP_NAME,
    MAX_HISTORY_MESSAGES,
    CHROMA_DIR,
)
from database import get_chat_history, save_chat
from export_chat import chat_to_pdf, chat_to_text
from feedback import record_feedback
from generator import stream_model
from output_guard import sanitize_output
from rate_limiter import allow_request
from retriever import (
    build_context,
    get_document_overview_samples,
    get_retrieval_info,
    get_sources,
    retrieve_documents,
)
from security import security_check


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="centered",
)

@st.cache_resource
def initialize_chroma():
    chroma_path= Path(CHROMA_DIR)

    if not chroma_path.exists() or not any(chroma_path.iterdir()):

       create_vector_database(reset=True)
    return True
    
try: 
    initialize_chroma()
    
except Exception as exc:
     st.error(f"Failed to initialize ChromaDB: {exc}")
     st.stop()
    
st.title(f"🤖 {APP_NAME}")
st.caption(APP_DESCRIPTION)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def is_greeting(question):
    """
    أسئلة الترحيب/التحية البسيطة وأسئله الشكر والمدح جاوب عليهم بطريقه لطيفه. لا تدخل هنا أي أسئلة عن الهوية
    أو المطور أو محتوى النظام، عشان تلك الأسئلة تمر على retrieve_documents
    العادي وترجع 'لا يوجد سياق كافٍ' تلقائياً لو مفيش معلومة عنها في الـ PDF.
    """
    text = " ".join(question.strip().lower().split())
    normalized = (
        text.replace("؟", "")
        .replace("?", "")
        .replace("!", "")
        .strip()
    )

    greetings = {
        "عامل ايه", "عامل إيه", "عامل اية", "ايه الاخبار", "ايه الدنيا","wie geht's","Wie geht's?",
        "عامل ازاي", "عامل إزاي", "ازيك يا صديق", "ازيك يا صديقى","يا صاحبى",
        "اهلا", "أهلا", "اهلا بيك", "أهلا بيك","صاحبى","ابو الصحاب",
        "السلام عليكم", "سلام عليكم", "سلام عليك","come va","apa kabarmu",
        "صباح الخير", "مساء الخير", "good morning", "Guten Morgen", "ogenki","kakdela",
        "ازيك", "إزيك", "how are you", "How are you",
        "hi", "hello", "hey", "welcom", "Welcom", "Hi", "hallo","Super",
        "good","goog","good work","super","ok" ,"Ok","Super", "nice", "very good", "thogenkianks", "thank you", "عمل رائع", "شكرا", "بالتوفيق", "ممتاز","عمل جيد","Nice",
    }

    return normalized in greetings


def is_content_scope_question(question):
    text = " ".join(question.strip().lower().split())
    normalized = (
        text.replace("؟", "")
        .replace("?", "")
        .replace("!", "")
        .strip()
    )

    scope_keywords = [
        "عن ايه المحتوى", "عن إيه", "المحتوى عن اية",
        "المحتوى عن", "المادة عن", "الماده عن","Was ist der Inhalt?","Was Inhalt",
        "بتفهم في ايه", "بتفهم في إيه","concept",
        "الاسئله اللي", "الأسئلة التي", "اسئله ممكن",
        "المواضيع اللي", "المواضيع التي","der kurs","Kurs",
        "ايه المحتوى", "إيه المحتوى","topic", "Inhalt","inhalt",
        "what topics", "what can i ask", "what is this about", "topics", "Topics","topic course",
    ]

    return any(keyword in normalized for keyword in scope_keywords)


CONTENT_SCOPE_ANSWER = (
    "material : SQL , NoSQL , Indexing , Normalization , "
    "Transaction ,Relation ,SCHEMA, Question & Answer , "
    "Primary Key ,Forign Key, Erd"
)


def is_full_summary_question(question):
    text = " ".join(question.strip().lower().split())
    normalized = (
        text.replace("؟", "").replace("?", "").replace("!", "").strip()
    )

    summary_keywords = [
        "ملخص", " لخص الماده باختصار", "تلخيص", "هات ملخص الماده", "summarize course", 
        "sumarize","zusammenfassung","Zusammenfassung",
        "ملخص للماده", "ملخص المنهج", "ملخص الكورس", "summary course", "summary of the course", 
        "summarize","summarise",
        "summarize material", "give me a summary", "overview of the material", 
        "overview material", "summarise course", "overview course",
    ]
    return any(keyword in normalized for keyword in summary_keywords)


def is_example_request(question):
    text = " ".join(question.strip().lower().split())
    normalized = (
        text.replace("؟", "").replace("?", "").replace("!", "").strip()
    )

    example_markers = [

        "امثله", "أمثلة", "مثال", "question" , "questions" , "Question&Answer", "Question & Answer","scenario",
        "امثلة","ich hatte gerne beispiele","beispiele","Beispiele","command","commands","code","give me a scenario","give me sql",
        "سيناريو", "سيناريوهات", "هات اسئله واجابات","هات اسئله امتحان والاجابه","many to many","one to many","give me commands",
        "هات اسئله امتحانات","هات عدد اسئله من اسئله الامتحان","اكتب كود","هات اوامر","هات علاقه","sql اكتب","give me command",
        "اسئله واجوبه", "أسئلة وأجوبة", "سؤال وجواب","هات عدد اسئله من اسئله امتحانات","write sql","one to one","Query",
        "هات عدد اسئله من اسئله الامتحانات", "fragen", "Fragen", "Question","relation","sql command","write a query",
    ]

    return any(marker in normalized for marker in example_markers)


def display_sources(sources, retrieval_info=None):
    if not sources and not retrieval_info:
        return

    with st.expander("📚 Sources & retrieval details"):
        if sources:
            st.markdown("**Sources**")
            for source in sources:
                st.write(f"- {source}")

        if retrieval_info:
            variants = retrieval_info.get("query_variants", [])
            if variants:
                st.markdown("**Search variants used**")
                for variant in variants:
                    st.code(variant, language=None)


for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(
                message.get("sources", []),
                message.get("retrieval_info"),
            )

            if message.get("question"):
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("👍", key=f"up_{index}"):
                        record_feedback(
                            question=message["question"],
                            answer=message["content"],
                            feedback="up",
                            sources=message.get("sources", []),
                            session_id=st.session_state.session_id,
                        )
                        st.success("Thanks for the feedback.")

                with col2:
                    if st.button("👎", key=f"down_{index}"):
                        record_feedback(
                            question=message["question"],
                            answer=message["content"],
                            feedback="down",
                            sources=message.get("sources", []),
                            session_id=st.session_state.session_id,
                        )
                        st.info("Feedback recorded.")


question = st.chat_input("Ask a question about the knowledge base...")

if question:
    client_id = st.session_state.session_id

    if not allow_request(client_id):
        st.error(
            "Too many requests. Please wait a little before sending another question."
        )
        st.stop()

    allowed, security_message = security_check(question)

    if not allowed:
        st.error(security_message)
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    history_for_rag = st.session_state.messages[:-1][-MAX_HISTORY_MESSAGES:]

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources = []
        retrieval_info = {}

        if is_greeting(question):
            placeholder = st.empty()
            chunks = []

            for chunk in stream_model(
                question=question,
                context="",
                history=history_for_rag,
            ):
                chunks.append(chunk)
                placeholder.markdown("".join(chunks) + "▌")

            answer = sanitize_output("".join(chunks))
            placeholder.markdown(answer)

        elif is_content_scope_question(question):
            answer = CONTENT_SCOPE_ANSWER
            st.markdown(answer)

        elif is_full_summary_question(question):
            documents = get_document_overview_samples(max_chunks=10)
            context = build_context(documents)
            sources = get_sources(documents)

            placeholder = st.empty()
            chunks = []

            for chunk in stream_model(
                question=(
                    "لخّص المحتوى العام للمادة الدراسية التالية في نقاط "
                    "رئيسية واضحة، بناءً فقط على المقتطفات الموزعة أدناه "
                    "التي تمثل عينة من الكتاب كامل."
                ),
                context=context,
                history=history_for_rag,
            ):
                chunks.append(chunk)
                placeholder.markdown("".join(chunks) + "▌")

            answer = sanitize_output("".join(chunks))
            placeholder.markdown(answer)

        elif is_example_request(question):
            documents = get_document_overview_samples(max_chunks=6)
            context = build_context(documents)
            sources = get_sources(documents)

            placeholder = st.empty()
            chunks = []

            for chunk in stream_model(

               question=(
                 f"{question}\n\n"
                 "ملاحظة: إذا كنت قد أعطيت أمثلة مشابهة في الرد السابق ضمن "
                 "سياق المحادثة، قدّم أمثلة مختلفة تمامًا هذه المرة ولا تكرر "
                 "نفس الأمثلة أو الصياغة."
                ),

                context=context,
                history=history_for_rag,
            ):
                chunks.append(chunk)
                placeholder.markdown("".join(chunks) + "▌")

            answer = sanitize_output("".join(chunks))
            placeholder.markdown(answer)

        else:
            documents = retrieve_documents(
                question=question,
                history=history_for_rag,
            )

            context = build_context(documents)
            sources = get_sources(documents)
            retrieval_info = get_retrieval_info(documents)

            if not context.strip():
                answer = (
                    "The current knowledge base does not contain enough relevant "
                    "information to answer this question."
                )
                st.markdown(answer)
            else:
                placeholder = st.empty()
                chunks = []

                for chunk in stream_model(
                    question=question,
                    context=context,
                    history=history_for_rag,
                ):
                    chunks.append(chunk)
                    placeholder.markdown("".join(chunks) + "▌")

                answer = sanitize_output("".join(chunks))
                placeholder.markdown(answer)

        display_sources(sources, retrieval_info)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_info": retrieval_info,
            "question": question,
        }
    )

    save_chat(
        question=question,
        answer=answer,
        sources=sources,
        session_id=st.session_state.session_id,
    )


with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.download_button(
        "📄 Export TXT",
        data=chat_to_text(st.session_state.messages),
        file_name="rag_chat.txt",
        mime="text/plain",
    )

    st.download_button(
        "📕 Export PDF",
        data=chat_to_pdf(st.session_state.messages),
        file_name="rag_chat.pdf",
        mime="application/pdf",
    )

    st.divider()
    st.subheader("Admin")

    admin_password = st.text_input(
        "Admin password",
        type="password",
    )

    if (
        ADMIN_PASSWORD
        and admin_password
        and admin_password == ADMIN_PASSWORD
    ):
        st.success("Admin authenticated.")

        history = get_chat_history(limit=100)

        if history:
            for item in history:
                st.write(item)
        else:
            st.info("No stored chat history.")

