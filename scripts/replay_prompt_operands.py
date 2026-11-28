"""Compare archived and current prompt-operand grounding without running a model or opening a DB."""

import argparse
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from crewforge.tools.arithmetic import DeterministicArithmeticTool, build_provenance_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_trace", type=Path)
    args = parser.parse_args()
    trace = json.loads(args.full_trace.read_text())
    catalog = build_provenance_catalog(original_prompt=trace["prompt"])
    transitions = Counter()
    for record in trace["tool_executions"]:
        for name, detail in record["provenance_details"].items():
            if detail.get("kind") != "prompt_span":
                continue
            _, current = DeterministicArithmeticTool.check_inputs_grounded(
                variables={name: Decimal(detail["value"])},
                input_provenance_refs={name: detail["ref"]}, catalog=catalog,
                operand_units={name: detail["units"]},
                source_quotes={name: detail.get("source_quote")},
            )
            transitions[detail["code"], current[name]["code"]] += 1
    print(json.dumps([
        {"recorded": old, "replayed": new, "operands": n}
        for (old, new), n in sorted(transitions.items())
    ], indent=2))


if __name__ == "__main__":
    main()
