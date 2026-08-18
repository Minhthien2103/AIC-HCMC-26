import csv
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402
from src.online_pipeline.object_filter import ObjectFilter  # noqa: E402
from src.online_pipeline.query_encoder import QueryEncoder  # noqa: E402
from src.online_pipeline.retrieval import RetrievalEngine  # noqa: E402
from src.online_pipeline.semantic_object_filter import SemanticObjectFilter  # noqa: E402
from src.online_pipeline.vlm_pipeline import VLMPipeline  # noqa: E402
from src.submission.formatting import format_kis_row, format_qa_row, format_trake_row  # noqa: E402
from src.tasks.kis_t import KIStask  # noqa: E402
from src.tasks.trake import TrakeTask  # noqa: E402
from src.tasks.vqa import VQATask  # noqa: E402

st.set_page_config(page_title="AI Challenge 2026", layout="wide")
st.title("AI Challenge HCMC 2026 - Retrieval System")


def resolve_keyframe_path(result: dict) -> str:
    return str(config.keyframe_path(result["video_id"], result["keyframe_name"]))


def rows_to_csv(rows: list[list[object]]) -> str:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


@st.cache_resource
def load_dependencies():
    encoder = QueryEncoder(config.CLIP_MODEL_NAME, config.CLIP_PRETRAINED)
    retriever = RetrievalEngine(config.FAISS_INDEX_PATH, config.METADATA_PATH)
    object_filter = ObjectFilter(config.OBJECTS_PATH)
    vlm = VLMPipeline(model_name="Qwen/Qwen2-VL-7B-Instruct")
    semantic_filter = SemanticObjectFilter(object_filter.get_all_labels())
    return (
        KIStask(encoder, retriever, object_filter),
        VQATask(encoder, retriever, object_filter, vlm, semantic_filter),
        TrakeTask(encoder, retriever, vlm),
    )


kis_task, vqa_task, trake_task = load_dependencies()
st.sidebar.header("Controls")
task = st.sidebar.selectbox("Task", ["KIS-T", "VQA", "TRAKE"])
top_k = st.sidebar.slider("Top K", 10, 100, 50, step=10)
object_filter_input = st.sidebar.text_input("Object Filter (comma separated)", placeholder="e.g. Person, Car")

if task == "KIS-T":
    st.header("Textual Known Item Search - KIS-T")
    query = st.text_area("Query", placeholder="Describe the scene")
    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching..."):
                results = kis_task.execute(query, top_k=top_k, object_labels=object_filter_input)
            if not results:
                st.info("No results found.")
            else:
                if results[0].get("auto_extracted"):
                    st.info(f"Auto-extracted objects: {', '.join(results[0]['auto_extracted'])}")
                st.success(f"Found {len(results)} results.")
                output_rows = []
                cols = st.columns(3)
                for idx, result in enumerate(results):
                    try:
                        output_rows.append(format_kis_row(result))
                    except (KeyError, ValueError):
                        continue
                    with cols[idx % 3]:
                        path = resolve_keyframe_path(result)
                        if os.path.exists(path):
                            st.image(path, width="stretch")
                        else:
                            st.warning(f"Image not found: {path}")
                        st.markdown(f"**{result['video_id']}** | Frame: `{result['frame_id']}`")
                        st.markdown(f"Score: `{result.get('score', 0):.4f}`")
                        st.code(f"{result['video_id']}, {result['frame_id']}", language="text")
                if output_rows:
                    st.download_button("Download KIS CSV", rows_to_csv(output_rows), "kis.csv", "text/csv")

elif task == "VQA":
    st.header("Video Question Answering - VQA")
    question = st.text_area("Question", placeholder="What is happening in this scene?")
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing with Qwen2-VL..."):
                results, analysis = vqa_task.execute(question, top_k=top_k)
            with st.expander("VLM Query Analysis (JSON)", expanded=False):
                st.json(analysis)
            if not results:
                st.info("No candidates found.")
            else:
                valid_rows = []
                cols = st.columns(3)
                for idx, result in enumerate(results):
                    with cols[idx % 3]:
                        path = resolve_keyframe_path(result)
                        if os.path.exists(path):
                            st.image(path, width="stretch")
                        st.markdown(f"**{result['video_id']}** | Frame: `{result['frame_id']}`")
                        try:
                            row = format_qa_row(result)
                            valid_rows.append(row)
                            st.success(f"Answer: {row[2]}")
                        except ValueError as exc:
                            st.error(f"Invalid VLM answer: {exc}")
                            if result.get("answer_error"):
                                st.caption(result["answer_error"])
                st.success(f"Found {len(valid_rows)} valid submission rows from {len(results)} candidates.")
                if valid_rows:
                    st.download_button("Download VQA CSV", rows_to_csv(valid_rows), "qa.csv", "text/csv")

else:
    st.header("TRAKE - Temporal Event Sequence")
    video_desc = st.text_input("Overall Video Description", placeholder="Describe the video")
    st.caption("Optional manual events; leave blank to let Qwen2-VL extract them.")
    event_count = st.number_input("Number of manual events", min_value=0, max_value=10, value=3, step=1)
    event_inputs = [st.text_input(f"Event {idx + 1}") for idx in range(event_count)]
    manual_events = [event for event in event_inputs if event.strip()]
    if st.button("Search Sequence", type="primary"):
        if not video_desc.strip():
            st.warning("Please enter a video description.")
        elif event_count and len(manual_events) != event_count:
            st.warning("Fill all event fields or set the event count to zero for VLM extraction.")
        else:
            with st.spinner("Finding complete chronological sequences..."):
                results = trake_task.execute(video_desc, manual_events, top_videos=5, max_sequences=top_k)
            if not results:
                st.info("No complete chronological sequence found.")
            else:
                auto_events = results[0].get("auto_extracted_events")
                if auto_events:
                    st.info(f"VLM events: {', '.join(auto_events)}")
                rows = []
                for result in results:
                    try:
                        rows.append(format_trake_row(result))
                    except ValueError:
                        continue
                    st.success(f"Video: {result['video_id']} | Sequence score: {result['sequence_score']:.4f}")
                    columns = st.columns(len(result["events"]))
                    for idx, event_result in enumerate(result["events"]):
                        with columns[idx]:
                            path = resolve_keyframe_path(event_result)
                            if os.path.exists(path):
                                st.image(path, width="stretch")
                            st.caption(f"Event {idx + 1}: frame {event_result['frame_id']}")
                    st.code(", ".join(str(value) for value in rows[-1]), language="text")
                if rows:
                    st.download_button("Download TRAKE CSV", rows_to_csv(rows), "trake.csv", "text/csv")
