/*
* Script used in checklist.yml. 
* Verifies which workflows are available for each PR 
* and lists which ones have run and with which outcome, 
* as a checklist for reviewers before approving PRs.
*/
module.exports = async ({github, context, core}) => {

    const { COMMIT_HASH, ISSUE_NUMBER } = process.env

    // List all available workflows
    const workflow_pages = github.rest.actions.listRepoWorkflows({
        owner: context.repo.owner,
        repo: context.repo.repo
    })
    const workflows = await github.paginate(workflow_pages)

    // List all workflows runs for each of them
    for (const workflow of workflows) {
        const workflow_runs_pages = github.rest.actions.listRepoWorkflows({
            owner: context.repo.owner,
            repo: context.repo.repo,
            workflow_id: workflow.id
        })
        const workflow_runs = await github.paginate(workflow_runs_pages)

        // Get only the run that was executed against the latest commit, if any
        workflow.latest_run = null
        for (const workflow_run of workflow_runs) {
            if (workflow_run === COMMIT_HASH) {
                workflow.latest_run = workflow_run
            }
        }
    }

    console.log(workflows)

    // List all comments on this PR
    const comments_pages = octokit.rest.issues.listComments({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: ISSUE_NUMBER
    });
    const comments = await github.paginate(comments_pages)

    // Delete all comments from this bot
    for (const comment of comments) {
        if (comment.user.login === 'github-actions') {
                await octokit.rest.issues.deleteComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                comment_id: comment.id
            });
        }
    }

    // Create a new comment
    await github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: '✨ New message ✨'
    })

}