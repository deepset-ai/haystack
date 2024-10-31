Feature: Pipeline running

    Scenario Outline: Running a correct Pipeline
        Given a pipeline <kind>
        When I run the Pipeline
        Then it should return the expected result
        And components ran in the expected order

        Examples:
        | kind |
        | that has no components |
        | that is linear |
        | that is really complex with lots of components, forks, and loops |
        | that has a single component with a default input |
        | that has two loops of identical lengths |
        | that has two loops of different lengths |
        | that has a single loop with two conditional branches |
        | that has a component with dynamic inputs defined in init |
        | that has two branches that don't merge |
        | that has three branches that don't merge |
        | that has two branches that merge |
        | that has different combinations of branches that merge and do not merge |
        | that has two branches, one of which loops back |
        | that has a component with mutable input |
        | that has a component with mutable output sent to multiple inputs |
        | that has a greedy and variadic component after a component with default input |
        | that has components added in a different order from the order of execution |
        | that has a component with only default inputs |
        | that has a component with only default inputs as first to run and receives inputs from a loop |
        | that has multiple branches that merge into a component with a single variadic input |
        | that has multiple branches of different lengths that merge into a component with a single variadic input |
        | that is linear and returns intermediate outputs |
        | that has a loop and returns intermediate outputs from it |
        | that is linear and returns intermediate outputs from multiple sockets |
        | that has a component with default inputs that doesn't receive anything from its sender |
        | that has a component with default inputs that doesn't receive anything from its sender but receives input from user |
        | that has a loop and a component with default inputs that doesn't receive anything from its sender but receives input from user |
        | that has multiple components with only default inputs and are added in a different order from the order of execution |
        | that is linear with conditional branching and multiple joins |
        | that is a simple agent |
        | that has a variadic component that receives partial inputs |
        | that has an answer joiner variadic component |
        | that is linear and a component in the middle receives optional input from other components and input from the user |
        | that has a loop in the middle |
        | that has variadic component that receives a conditional input |
        | that has a string variadic component |

    Scenario Outline: Running a bad Pipeline
        Given a pipeline <kind>
        When I run the Pipeline
        Then it must have raised <exception>

        Examples:
        | kind | exception |
        | that has an infinite loop | PipelineMaxComponentRuns |
        | that has a component that doesn't return a dictionary | PipelineRuntimeError |
        | that has a cycle that would get it stuck | PipelineRuntimeError |
