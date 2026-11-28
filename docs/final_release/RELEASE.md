# Release and rollback

## Architecture

The planner decomposes the user's request into deliverables and schedules specialists. Independent tasks can run in parallel. The workers submit scoped candidate findings and grounded arithmetic, exchange findings and objections, and pass their versioned work to independent review. The lead reconciles open requirements. The release gate checks endorsements, assumptions and dependency lineage before rendering the answer. The single agent uses the same underlying calculator with its own normal refinement path.

The production image builds the React office and serves it on the same origin as the asynchronous MCP service. Public jobs use `/mcp`, while private workspace MCP uses `/mcp/workspace`. Dashboard APIs keep their existing browser authentication. Public and private databases have separate paths.

## Deploy the existing Railway service

1. Open the existing service and record the current active deployment ID, source commit, build settings, active replica count and mount paths. Retain the previous image and take an operator-approved volume snapshot. Keep one replica and the existing volume mounted at `/data`.
2. From the validated clean release checkout publish with `git push origin HEAD:refs/heads/codex/final-release`. Use the project's existing repository source and deploy this branch at the verified commit. Retain the existing public domain and service. Use the repository Dockerfile and set the health check to `/health`.
3. Keep `CREWFORGE_RELEASE_ENABLE_PAID_INFERENCE=false` and `CREWFORGE_ALLOW_PAID_RUNS=false`. Keep existing provider credentials and limits in server-side Railway variables. Retain `CREWFORGE_PUBLIC_MCP_DB_PATH` and `CREWFORGE_PUBLIC_MCP_BUDGET_DB_PATH` at their existing persistent paths. Set `CREWFORGE_DASHBOARD_DATA_DIR` to the existing private persistent location if there is one. The default for a new private workspace is `/data/dashboard`. Preserve existing Google/email/auth configuration. Railway supplies `RAILWAY_PUBLIC_DOMAIN`, `RAILWAY_GIT_COMMIT_SHA` and `PORT`. An existing custom hostname can be added through `CREWFORGE_ALLOWED_HOSTS`.
4. Startup runs the existing SQLite migrations in place. Check `GET /health` for the exact commit and paid-inference status. Verify frontend assets over HTTPS, unauthenticated rejection from private APIs, and anonymous public Streamable HTTP initialization and discovery. Use only a non-executing protocol check until the inference allocation is approved.

For rollback, select the previously recorded successful deployment in the same Railway service and use Rollback or Redeploy for that deployment. Keep the volume attached. Confirm that the rollback image can read the live schema. Restore the snapshot only if the schema needs restoration and an operator has approved the associated data loss. Verify the rollback commit against deployment metadata and repeat the health and discovery checks.

## Official Team Pack checklist

| Requirement | Evidence and status |
| --- | --- |
| No login or client credentials | PASS offline for public MCP. Private authentication remains in place. |
| Permanent public HTTPS | Previous service resolves over HTTPS. Verify the release deployment and commit independently. |
| No judge permission errors | PASS offline for anonymous initialization and discovery. Verify public invocation after deployment. |
| Streamable HTTP | PASS offline using the MCP SDK. |
| All tools work | Offline suite exercises submission, polling, results, traces, baseline, cancellation and errors. A real AI-assistant execution is pending paid approval. |
| Quick tool responses | Queue submission returns a job ID and supports polling. Production timing remains to be checked. |
| Google Form submission | PENDING OWNER ACTION. Attach the labelled materials when the form arrives. |
| Every repository shared | PENDING OWNER ACTION. Confirm links and private collaborator access if needed. |
| One mentor call booked and attended | PENDING OWNER ACTION. The booking link arrives separately. Missing attendance disqualifies the team. |
