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

    Scenario Outline: Running a bad Pipeline
        Given a pipeline <kind>
        When I run the Pipeline
        Then it must have raised <exception>

        Examples:
        | kind | exception |
        | that has an infinite loop | PipelineMaxLoops |
