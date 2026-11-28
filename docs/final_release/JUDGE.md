# Judge quick start

Use a real AI assistant that supports remote MCP. Add the verified CrewForge public URL as a Streamable HTTP server. The existing Railway URL is [CrewForge MCP](https://crewforge-production-1e4fcb21.up.railway.app/mcp). The active release commit must be confirmed before the demonstration.

After the owner approves and enables the judging allocation, ask the assistant to use CrewForge's `run_task` with a task of your choosing. Ask it to poll `get_task_status` until terminal, then retrieve `get_task_result` and `get_task_trace`. Have it show the final answer, sources, any open requirements, and the agent assignments or contributions recorded in the trace.

For a matched comparison, submit the same text and documents once using `run_single_agent`. Poll that ID and obtain its own result. Keep each result, including partial or failed responses. View model identity, calls, token usage and cost separately from answer quality. None of this requires a dashboard account or client provider key.
