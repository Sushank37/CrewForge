export type PendingMessage = { fingerprint: string; id: string };

export async function submissionKey(scope: string, content: unknown): Promise<string> {
  const payload = new TextEncoder().encode(JSON.stringify(content));
  const digest = await crypto.subtle.digest("SHA-256", payload);
  const fingerprint = Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
  const key = `crewforge.pending.${scope}`;
  const pending = JSON.parse(sessionStorage.getItem(key) || "null") as PendingMessage | null;
  if (pending?.fingerprint === fingerprint) return pending.id;
  const id = crypto.randomUUID();
  sessionStorage.setItem(key, JSON.stringify({ fingerprint, id }));
  return id;
}

export function acknowledgeSubmission(scope: string) {
  sessionStorage.removeItem(`crewforge.pending.${scope}`);
}

export function historicalAnswer(answer: string): string {
  // Historic replies retain their original records. Hide only stable release annotations.
  return answer
    .split("\n\n")
    .filter((block) => !/^\*\*(?:Calculations and Sources|Partial Result|Incomplete Tasks)\b/.test(block))
    .map((block) => block
      .replace(/^\*\*(?:Proposed Actions \(Advisory\)|Conditional \/ Qualified Statements):\*\*\s*/g, "")
      .replace(/^\*\*Unresolved \/ Withheld Items:\*\*/, "What I couldn’t verify")
      .replace(/ \[(?:advisory; )?clm_[\w]+@v\d+[^\]\n]*\]/g, "")
      .replace(/\(clm_[\w]+ v\d+\)/g, ""))
    .filter(Boolean).join("\n\n");
}
