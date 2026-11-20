"""Terminal chat.

    python -m app.cli                 # interactive chat
    python -m app.cli --debug         # also print the retrieved passages
    python -m app.cli -q "What is Gnani Prisma?"   # one question, then exit
"""

from __future__ import annotations

import argparse

from app.chatbot import GnaniChatbot
from app.retriever import Retriever


def print_reply(reply, debug: bool) -> None:
    if debug:
        print("\n[retrieved passages]")
        for i, c in enumerate(reply.retrieved, 1):
            print(f"  {i}. {c.title} :: {c.heading[:60]}  ({c.url})")
        print(f"[status={reply.status}, llm_called={reply.llm_called}]")
    print(f"\nBot: {reply.answer}")
    for s in reply.sources:
        print(f"     source: {s.title} - {s.url}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", help="ask one question and exit")
    ap.add_argument("--debug", action="store_true", help="show retrieved passages and status")
    args = ap.parse_args()

    bot = GnaniChatbot(Retriever.from_file())
    if args.question:
        print_reply(bot.ask(args.question), args.debug)
        return

    print("Gnani.ai assistant. Ask about Gnani AI (type 'exit' to quit).\n")
    history: list[dict] = []
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        reply = bot.ask(q, history)
        print_reply(reply, args.debug)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": reply.answer}]


if __name__ == "__main__":
    main()
