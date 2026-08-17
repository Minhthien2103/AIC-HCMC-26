import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.online_pipeline.query_encoder import QueryEncoder
from src.online_pipeline.retrieval import RetrievalEngine
from src.online_pipeline.object_filter import ObjectFilter
from src.online_pipeline.semantic_object_filter import SemanticObjectFilter
from src.online_pipeline.vlm_pipeline import VLMPipeline
from src.tasks.kis_t import KIStask
from src.tasks.vqa import VQATask
from src.tasks.trake import TrakeTask

st.set_page_config(page_title = "AI Challenge 2026", layout = "wide")

st.title("AI Challenge HCMC 2026 - Retrieval System")


def resolve_keyframe_path(res: dict) -> str:
    """Rebuild the keyframe path using current config so stale absolute paths (from a renamed project folder) still work."""
    return str(config.KEYFRAMES_DIR / res["video_id"] / (res["keyframe_name"] + ".jpg"))


@st.cache_resource
def load_dependencies():
    encoder = QueryEncoder(model_name = config.CLIP_MODEL_NAME, pretrained = config.CLIP_PRETRAINED)
    retriever = RetrievalEngine(index_path = config.FAISS_INDEX_PATH, metadata_path = config.METADATA_PATH)
    obj_filter = ObjectFilter(object_path = config.OBJECTS_PATH)
    vlm = VLMPipeline(model_name="Qwen/Qwen2-VL-7B-Instruct")

    sem_filter = SemanticObjectFilter(obj_filter.get_all_labels())
    kis_task = KIStask(encoder = encoder, retriever = retriever, object_filter = obj_filter)
    vqa_task = VQATask(encoder = encoder, retriever = retriever, object_filter = obj_filter, vlm_pipeline = vlm, semantic_filter = sem_filter)
    trake_task = TrakeTask(encoder = encoder, retriever = retriever, vlm_pipeline = vlm)

    return kis_task, vqa_task, trake_task

kis_task, vqa_task, trake_task = load_dependencies()


st.sidebar.header("Controls")
task = st.sidebar.selectbox("Task", ["KIS-T", "VQA", "TRAKE"])
top_k = st.sidebar.slider("Top K", 10, 100, 50, step = 10)
object_filter_input = st.sidebar.text_input("Object Filter (comma seperated)", placeholder = "e.g. Person, Car")


if task == "KIS-T":
    st.header("Textual Known Item Search - KIS-T")

    query = st.text_area("Query", placeholder = "Describe the scene")

    if st.button("Search", type = "primary"):
        if not query.strip():
            st.warning("Not has query yet")
        else:
            with st.spinner("Searching..."):
                results = kis_task.execute(query = query, top_k = top_k, object_labels = object_filter_input)

            if not results:
                st.info("No results found.")
            else:
                if "auto_extracted" in results[0] and results[0]["auto_extracted"]:
                    st.info(f"**Auto-Extracted Objects:** {', '.join(results[0]['auto_extracted'])}")
                
                st.success(f"Found {len(results)} results.")
                cols = st.columns(3)

                for idx, res in enumerate(results):
                    col = cols[idx % 3]

                    with col:
                        v_id = res["video_id"]
                        f_id = res["frame_id"]
                        score = res["score"]
                        k_path = resolve_keyframe_path(res)

                        try:
                            if os.path.exists(k_path):
                                st.image(k_path, width='stretch')
                            else:
                                st.warning(f"Image not found: {k_path}")

                        except Exception as e:
                            pass

                        st.markdown(f"**{v_id}** | Frame: `{f_id}`")
                        st.markdown(f"Score: `{score:.4f}`")

                        if "matched_objects" in res and res["matched_objects"]:
                            st.caption(f"Objects: {', '.join(res['matched_objects'])}")
                            
                        st.code(f"{v_id}, {f_id}", language = "text")

elif task == "VQA":
    st.header("Video Question Answering - VQA")
    
    question = st.text_area("Question", placeholder = "What is happening in this scene?")
    
    if st.button("Ask", type = "primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing..."):
                results, analysis = vqa_task.execute(question, top_k = top_k)
                
            with st.expander("VLM Query Analysis (JSON)", expanded=False):
                st.json(analysis)
                
            if not results:
                st.info("No results found.")
            else:
                st.success(f"Found {len(results)} candidate frames.")
                cols = st.columns(3)
                for idx, res in enumerate(results):
                    col = cols[idx % 3]
                    with col:
                        vid = res['video_id']
                        fid = res['frame_id']
                        k_path = resolve_keyframe_path(res)
                        if os.path.exists(k_path):
                            st.image(k_path, width='stretch')

                        st.markdown(f"**{vid}** | Frame: `{fid}`")
                        st.error(f"**Answer:** {res['answer']}")

elif task == "TRAKE":
    st.header("TRAKE - Temporal Event Sequence")
    
    video_desc = st.text_input("Overall Video Description (VLM will automatically extract events!)", placeholder = "A cooking tutorial where a chef chops onions, then puts them in a pan...")
    
    st.markdown("---")
    st.caption("Optional: Override the VLM by manually specifying the events below:")
    col1, col2, col3 = st.columns(3)
    with col1:
        e1 = st.text_input("Event 1 (Optional)", placeholder = "Chef chops onions")
    with col2:
        e2 = st.text_input("Event 2 (Optional)", placeholder = "Chef puts onions in pan")
    with col3:
        e3 = st.text_input("Event 3 (Optional)", placeholder = "Chef serves the dish")
        
    manual_events = [e for e in [e1, e2, e3] if e.strip()]
    
    if st.button("Search Sequence", type = "primary"):
        if not video_desc.strip():
            st.warning("Please enter a video description.")
        else:
            with st.spinner("Analyzing description and searching for sequence..."):
                results = trake_task.execute(video_desc, manual_events, top_videos = 5)
                
            if not results:
                st.info("No sequence found.")
            else:
                # If events were auto-extracted, show them
                if results[0].get("auto_extracted_events"):
                    extracted = results[0]["auto_extracted_events"]
                    st.info(f"**VLM Auto-Extracted Events:** {', '.join(extracted)}")
                    # Use the extracted events for rendering columns
                    events_to_display = extracted
                else:
                    events_to_display = manual_events
                    
                for res in results:
                    vid = res["video_id"]
                    is_valid = res["is_valid_sequence"]
                    
                    if is_valid:
                        st.success(f"Video: {vid} | Overall Score: {res['video_score']:.4f} (Valid Chronological Sequence)")
                    else:
                        st.warning(f"Video: {vid} | Overall Score: {res['video_score']:.4f} (Events are OUT OF ORDER)")
                        
                    event_cols = st.columns(len(events_to_display))
                    frame_ids = []
                    
                    for idx, e_res in enumerate(res["events"]):
                        with event_cols[idx]:
                            st.caption(f"Event {idx+1}")
                            k_path = resolve_keyframe_path(e_res)
                            fid = e_res['frame_id']
                            frame_ids.append(str(fid))
                            
                            if os.path.exists(k_path):
                                st.image(k_path, width='stretch')
                                
                            st.markdown(f"Frame: `{fid}` | Score: `{e_res['score']:.4f}`")
                            
                    st.code(f"{vid}, " + ", ".join(frame_ids), language="text")
                    
else:
    st.info(f"{task} is under constructed")




