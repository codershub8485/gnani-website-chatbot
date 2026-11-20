"""End-to-end evaluation against the real model (needs ANTHROPIC_API_KEY; costs a few cents).

    python -m eval.run_eval

For every question in eval/questions.jsonl it checks:
  * expect=answer : the bot answered (status "answered") and, if given, every
                    must_mention phrase appears in the answer (a light groundedness check);
  * expect=refuse : the bot did NOT answer (out_of_scope / not_found / blocked).
Writes per-question results to eval/last_run.json and prints a summary.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from app.chatbot import GnaniChatbot
from app.retriever import Retriever

HERE = Path(__file__).parent
REFUSALS = {"out_of_scope", "not_found", "blocked"}


def main() -> int:
    cases = [json.loads(l) for l in (HERE / "questions.jsonl").read_text().splitlines() if l.strip()]
    bot = GnaniChatbot(Retriever.from_file())
    results, passed = [], 0

    for case in cases:
        t0 = time.time()
        reply = bot.ask(case["question"])
        secs = time.time() - t0
        if reply.status == "error":
            print(f"ERROR: {reply.answer}")
            return 2
        if case["expect"] == "answer":
            missing = [m for m in case.get("must_mention", []) if m.lower() not in reply.answer.lower()]
            ok = reply.status == "answered" and not missing
            note = f"missing {missing}" if missing else ""
        else:
            ok = reply.status in REFUSALS
            note = ""
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  [{case['expect']:6}] -> {reply.status:12} {secs:4.1f}s  {case['question'][:70]} {note}")
        results.append({**case, "status": reply.status, "answer": reply.answer, "ok": ok,
                        "sources": [s.url for s in reply.sources], "seconds": round(secs, 1)})

    (HERE / "last_run.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    n_ans = sum(c["expect"] == "answer" for c in cases)
    ok_ans = sum(r["ok"] for r in results if r["expect"] == "answer")
    ok_ref = sum(r["ok"] for r in results if r["expect"] == "refuse")
    print(f"\nIn-scope answered correctly : {ok_ans}/{n_ans}")
    print(f"Out-of-scope refused        : {ok_ref}/{len(cases) - n_ans}")
    print(f"Overall                     : {passed}/{len(cases)}  (details in eval/last_run.json)")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
