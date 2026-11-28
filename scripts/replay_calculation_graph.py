"""Offline counterfactual replay of immutable exported arithmetic requests in a temporary DB."""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from crewforge.schemas import ArithmeticRequest, ArithmeticVariableBinding, ClaimStatus, RunTrace
from crewforge.storage.database import Database
from crewforge.tools.arithmetic import DeterministicArithmeticTool, build_provenance_catalog


def original_request(record):
    return ArithmeticRequest(
        expression=record.expression,
        proposed_result=record.proposed_result,
        units=record.units,
        assumptions=record.assumptions,
        variable_bindings=[
            ArithmeticVariableBinding(
                name=name,
                value=value,
                provenance_ref=record.input_provenance_refs[name],
                units=record.provenance_details[name]["units"],
                source_quote=record.provenance_details[name].get("source_quote"),
            )
            for name, value in record.variables.items()
        ],
    )


async def replay(trace, database):
    """Rebuild the proposal-time catalog. Never import final verdicts as operand authority."""
    await database.create_run(trace.run_id, "network", trace.prompt, trace.config)
    await database.save_execution_plan(trace.plan)
    tool = DeterministicArithmeticTool(database)
    owners = {tid: c for c in trace.claims for tid in c.tool_execution_ids}
    records, claims, rows = [], [], []
    for source in trace.tool_executions:
        catalog = build_provenance_catalog(
            original_prompt=trace.prompt,
            verified_evidence=[e for e in trace.evidence if e.created_at <= source.created_at],
            all_claims=claims,
            tool_records=records,
            run_id=trace.run_id,
        )
        for detail in source.provenance_details.values():
            # Same-response aliases are transient. The export records their actual resolution.
            if (
                detail.get("kind") == "tool_result"
                and detail["resolved_id"] in catalog.targets_by_key
            ):
                catalog.targets_by_key[detail["ref"]] = catalog.targets_by_key[
                    detail["resolved_id"]
                ]
        invalid = [
            ref
            for ref in source.provenance_details.get("__claim_premises__", {}).get("values", [])
            if ref not in catalog.targets_by_key or not catalog.targets_by_key[ref].is_trusted
        ]
        rec = await tool.execute(
            run_id=trace.run_id,
            actor=source.actor,
            request=original_request(source),
            catalog=catalog,
            invalid_premise_refs=invalid,
        )
        records.append(rec.model_copy(update={"tool_execution_id": source.tool_execution_id}))
        claims.append(
            owners[source.tool_execution_id].model_copy(update={"status": ClaimStatus.ASSERTED})
        )
        rows.append(
            {
                "id": source.tool_execution_id,
                "matches": rec.matches_proposed,
                "grounded": rec.inputs_grounded,
                "claim": owners[source.tool_execution_id].claim_id,
                "premises": owners[source.tool_execution_id].depends_on_claim_ids,
                "failures": {
                    k: v["code"] for k, v in rec.provenance_details.items() if not v["grounded"]
                },
            }
        )
    return rows


async def main(path):
    trace = RunTrace.model_validate_json(path.read_text())
    with TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "replay.db")
        await db.initialize()
        try:
            rows = await replay(trace, db)
        finally:
            await db.close()
    print(
        json.dumps(
            {
                "records": rows,
                "matches": sum(r["matches"] is True for r in rows),
                "grounded": sum(r["grounded"] for r in rows),
                "failures": Counter(c for r in rows for c in r["failures"].values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_trace", type=Path)
    asyncio.run(main(parser.parse_args().full_trace))
