from __future__ import annotations

from customer_service.services.chat import CustomerService


def main() -> None:
    service = CustomerService()
    session_id = "demo-session"
    print("Multi-agent customer service demo. Type 'exit' to quit.")
    while True:
        user_input = input("\nUser> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        result = service.chat(user_message=user_input, session_id=session_id, user_id="demo-user")
        print(f"Assistant> {result['final_response']}")
        print(f"[intent={result.get('intent')} agent={result.get('assigned_agent')} score={result.get('quality_score')}]")


if __name__ == "__main__":
    main()
