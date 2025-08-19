from pathlib import Path

from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator

snapshots_dir = "snapshots_simple"


def create_test_pipeline():
    # Create a simple prompt builder
    prompt_template = """
    Answer the following question: {{question}}

    Context: {{context}}

    Answer:
    """

    prompt_builder = PromptBuilder(template=prompt_template)

    # Create a mock LLM generator (you'll need to replace with your actual API key)
    llm = OpenAIGenerator()

    # Create pipeline
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)
    pipeline.connect("prompt_builder", "llm")

    return pipeline


def test_automatic_snapshots():
    pipeline = create_test_pipeline()
    test_data = {
        "prompt_builder": {"question": "What is the capital of France?", "context": "France is a country in Europe."}
    }

    print("Running pipeline with automatic state persistence...")
    results = pipeline.run(data=test_data, state_persistence=True, state_persistence_path=snapshots_dir)

    print("Pipeline completed successfully!")
    print(f"Results: {results}")

    # Check if snapshot files were created
    snapshot_files = list(Path(snapshots_dir).glob("*.json"))
    print(f"\nSnapshot files created: {len(snapshot_files)}")

    # resume from each snapshot and print details
    for snapshot_file in snapshot_files:
        print(f"  - {snapshot_file.name}")
        from haystack.core.pipeline.breakpoint import load_pipeline_snapshot

        try:
            snapshot = load_pipeline_snapshot(snapshot_file)
            print(f"    Component: {snapshot.break_point.component_name}")
            print(f"    Visit count: {snapshot.break_point.visit_count}")
            resumed_results = pipeline.run(data={}, pipeline_snapshot=snapshot)
            print(f"    Resumed results: {resumed_results}")
        except Exception as e:
            print(f"    Error loading snapshot: {e}")


if __name__ == "__main__":
    test_automatic_snapshots()
