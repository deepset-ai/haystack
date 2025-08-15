from pathlib import Path

from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses.breakpoints import Breakpoint

output_path = "snapshots"


# Create a simple RAG pipeline
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
    # Create the pipeline
    pipeline = create_test_pipeline()

    # Test data
    test_data = {
        "prompt_builder": {"question": "What is the capital of France?", "context": "France is a country in Europe."}
    }

    break_point = Breakpoint(component_name="prompt_builder", visit_count=0, snapshot_file_path=f"{output_path}")

    set_break_point = False

    # Run pipeline with automatic state persistence
    print("Running pipeline with automatic state persistence...")
    results = pipeline.run(
        data=test_data,
        break_point=break_point if set_break_point else None,
        state_persistence=True,
        state_persistence_path=output_path,
    )

    print("Pipeline completed successfully!")
    print(f"Results: {results}")

    # Check if snapshot files were created
    snapshot_files = list(Path(output_path).glob("*.json"))
    print(f"\nSnapshot files created: {len(snapshot_files)}")

    for snapshot_file in snapshot_files:
        print(f"  - {snapshot_file.name}")
        from haystack.core.pipeline.breakpoint import load_pipeline_snapshot

        try:
            snapshot = load_pipeline_snapshot(snapshot_file)
            print(f"    Component: {snapshot.break_point.component_name}")
            print(f"    Visit count: {snapshot.break_point.visit_count}")
        except Exception as e:
            print(f"    Error loading snapshot: {e}")


def run_from_snapshots():
    # Create the pipeline
    pipeline = create_test_pipeline()

    # Test resuming from a snapshot (if any were created)
    snapshot_files = list(Path(output_path).glob("*.json"))
    if snapshot_files:
        print("\nTesting resume from snapshot...")

        # Load the first snapshot
        from haystack.core.pipeline.breakpoint import load_pipeline_snapshot

        snapshot = load_pipeline_snapshot("snapshots/llm_2025_08_15_16_13_36.json")
        # print(snapshot_files[0].name)

        try:
            # Resume pipeline from snapshot
            resumed_results = pipeline.run(
                data={},  # Empty data since we're resuming
                pipeline_snapshot=snapshot,
            )
            print("Pipeline resumed successfully!")
            print(f"Resumed results: {resumed_results}")
        except Exception as e:
            print(f"Resume failed: {e}")


if __name__ == "__main__":
    # test_automatic_snapshots()
    run_from_snapshots()
