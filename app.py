import os

import streamlit as st

from evidence import DOMAINS, EXPERTS, MODES, Result, benchmark, load_evidence, saved_result, score_answer
from inference import ADAPTER_FILES, MODELS, ROUTER_FILES, MythoEngine

st.set_page_config(page_title="Mytho-MoE | Mythology QA", page_icon="📖", layout="wide")


@st.cache_data(show_spinner=False)
def paper_data():
    records, routing = load_evidence()
    return records, routing, benchmark(records)


@st.cache_resource(show_spinner=False)
def engine():
    return MythoEngine()


def remote_answer(space, query, modes):
    from gradio_client import Client

    client = Client(space, token=os.environ.get("MYTHO_HF_TOKEN"), verbose=False,
                    download_files=False, analytics_enabled=False)
    payload = client.submit(query, modes, api_name="/answer").result(timeout=240)
    if not isinstance(payload, dict) or set(payload) != set(modes):
        raise ValueError("The inference service returned an incomplete result.")
    return {mode: Result(**payload[mode]) for mode in modes}


def percent(value):
    return "—" if value is None else f"{value:.2%}"


def show_result(mode, result, expected, references, saved, summary, routing):
    with st.container(border=True):
        st.subheader(MODES[mode])
        if result.expert:
            st.markdown(f"**{DOMAINS[result.expert]} · {EXPERTS[result.expert]}**")
        else:
            st.caption("Backbone only · routing bypassed")

        st.markdown("**Answer**")
        if result.answer:
            st.text(result.answer)
        else:
            st.info("The model returned an empty answer.")

        if result.probabilities:
            top_score = result.probabilities[result.expert]
            route, confidence = st.columns(2)
            route.metric("Router confidence", percent(top_score), help="Top-1 softmax score for the selected domain.")
            route_correct = "Correct" if result.expert == expected else "Incorrect"
            confidence.metric("Route vs. expected", route_correct if expected else "—")
            for domain, probability in result.probabilities.items():
                st.progress(probability, text=f"{DOMAINS[domain]} · {percent(probability)}")

        likelihood, exact, f1 = st.columns(3)
        likelihood.metric(
            "Generation likelihood", percent(result.generation_likelihood),
            help="Geometric mean of the model probabilities of generated tokens, before text cleanup and excluding stop tokens.",
        )
        em, token_f1 = score_answer(result.answer, references) if references else (None, None)
        exact.metric("Exact match", {1.0: "Match", 0.0: "No match"}.get(em, "—"))
        f1.metric("Token F1", percent(token_f1))
        if saved:
            st.caption("The archived output has no generated-token probabilities.")
        if result.router_truncated:
            st.caption("The router used the first 256 query tokens.")
        if result.hit_generation_limit:
            st.caption("The answer reached the 256-token generation limit.")
        if mode in summary:
            with st.expander("Matched benchmark accuracy"):
                if mode in routing:
                    accuracy, count = routing[mode]
                    st.write(f"Routing accuracy · **{percent(accuracy)}** across {count:,} labeled queries")
                st.write(f"Configuration exact match · **{percent(summary[mode]['exact_match'])}** "
                         f"across {summary[mode]['n']:,} answers")
                if result.expert:
                    expert = summary[result.expert]
                    st.write(f"{DOMAINS[result.expert]} expert exact match · **{percent(expert['exact_match'])}** "
                             f"on {expert['n']:,} questions from its own domain")
        with st.expander("Run details"):
            details = [f"Generated tokens · {result.generated_tokens}"]
            if result.generation_seconds is not None:
                details.append(f"Generation · {result.generation_seconds:.2f} s")
            if result.router_seconds is not None:
                details.append(f"Shared routing pass · {result.router_seconds:.2f} s")
            if result.probabilities:
                margin = abs(result.probabilities["hvm"] - result.probabilities["rama"])
                details.append(f"Router score margin · {margin:.2%}")
                details.append("Expert adapter weight · 1.0")
            st.write("  \n".join(details))


def show_benchmarks(summary, routing):
    st.subheader("Matched short-answer benchmark")
    st.caption("Paper Table 4 · 2,000 answers and 1,991 unique routing queries")
    st.table([
        {"Configuration": label, "Routing accuracy": percent(routing.get(mode, (None,))[0]),
         "Answer exact match": percent(summary[mode]["exact_match"]), "Token F1": percent(summary[mode]["token_f1"])}
        for mode, label in MODES.items()
    ])
    st.subheader("Fixed-expert performance by domain")
    st.caption("Matched short-answer benchmark · 1,000 questions from each specialist’s domain")
    st.table([
        {"Expert": DOMAINS[domain], "Evaluation questions": summary[domain]["n"],
         "Answer exact match": percent(summary[domain]["exact_match"]), "Token F1": percent(summary[domain]["token_f1"])}
        for domain in DOMAINS
    ])


st.title("Mytho-MoE")
st.markdown("Compare how Qwen3-4B and two expert routers answer questions about the Harivamsha and the Ramayana.")

try:
    records, routing, summary = paper_data()
except (OSError, ValueError, KeyError) as error:
    st.warning(f"Archived benchmark data could not be loaded. {error}")
    records, routing, summary = [], {}, {}

qa, paper = st.tabs(["Question lab", "Paper benchmarks"])
with paper:
    if summary:
        show_benchmarks(summary, routing)

with qa:
    space = os.environ.get("MYTHO_SPACE")
    local_models = all((MODELS / filename).is_file()
                       for filename, _ in (*ROUTER_FILES.values(), *ADAPTER_FILES.values()))
    sources = ["Live inference", "Saved benchmark"] if space or local_models else ["Saved benchmark"]
    source = st.radio("Run source", sources, horizontal=True, key="source")
    if len(sources) == 1:
        st.caption("Live questions become available after the GPU service is connected.")
    saved = source == "Saved benchmark"
    record = None
    if saved and records:
        st.caption("Replay recorded predictions from the paper’s matched short-answer benchmark.")
        record = st.selectbox(
            "Benchmark question", range(len(records)), index=1 if len(records) > 1 else 0,
            format_func=lambda i: f"{DOMAINS[records[i]['domain']]} · {records[i]['query']}",
        )
    elif not saved:
        st.caption("Live inference uses the paper’s checkpointed routers and LoRA experts. The first run loads Qwen3-4B.")

    with st.form("question"):
        if not saved:
            query = st.text_area("Question", placeholder="Who is Rama’s father in the Valmiki Ramayana?", max_chars=8000)
            with st.expander("Score against a reference"):
                expected = st.selectbox("Expected expert", [None, *DOMAINS],
                                        format_func=lambda domain: DOMAINS.get(domain, "Unspecified"))
                reference = st.text_area("Reference answer", max_chars=8000,
                                         help="Exact match and token F1 compare model output with this answer.")
        elif record is not None:
            row = records[record]
            query, expected = row["query"], row["domain"]

        modes = st.multiselect("Configurations", list(MODES), default=list(MODES), format_func=MODES.get)
        submitted = st.form_submit_button("Show saved predictions" if saved else "Run question", type="primary",
                                          disabled=saved and record is None)

    if submitted:
        st.session_state.pop("run", None)
        if not modes or not query.strip():
            st.error("Enter a question and select at least one configuration.")
        else:
            query = query.strip()
            references = row["gold"] if saved else ([reference.strip()] if reference.strip() else [])
            try:
                if saved:
                    outputs = {mode: saved_result(row, mode) for mode in modes}
                else:
                    with st.spinner("Loading checkpoints and running inference…"):
                        outputs = remote_answer(space, query, modes) if space else engine().answer(query, modes)
                st.session_state["run"] = (source, record, query, expected, references, outputs)
            except Exception as error:
                if space and not saved:
                    st.error("The GPU service could not finish this request. It may be starting up or at its usage limit.")
                else:
                    st.error(f"The run could not finish. {error}")
                if not saved and not space:
                    st.caption("For setup and checkpoint download instructions, see the web-app README.")

    run = st.session_state.get("run")
    if run and run[:2] == (source, record):
        _, _, answered_query, expert_label, references, outputs = run
        st.divider()
        st.subheader("Results")
        st.text(answered_query)
        st.caption("Router confidence measures domain preference. Generation likelihood measures token probabilities. Accuracy comes from reference comparisons or labeled benchmarks.")
        if expert_label:
            st.caption(f"Expected expert · {DOMAINS[expert_label]}")
        if references:
            with st.expander("Reference answer", expanded=True):
                for target in references:
                    st.text(target)
        elif not expert_label:
            st.caption("Add an expected expert and reference answer to score this question.")
        for column, (mode, result) in zip(st.columns(len(outputs)), outputs.items()):
            with column:
                show_result(mode, result, expert_label, references, saved, summary, routing)
