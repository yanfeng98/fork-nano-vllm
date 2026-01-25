from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.pretty import Pretty
import json
import datetime

complex_data = {
    "user_id": 1024,
    "username": "Antigravity_User",
    "is_active": True,
    "roles": ["admin", "developer", "editor"],
    "meta": {
        "created_at": datetime.datetime.now().isoformat(),
        "login_count": 42,
        "preferences": {
            "theme": "dark",
            "notifications": None,
            "retry_attempts": 3
        }
    },
    "history": [
        {"action": "login", "timestamp": 1700000000},
        {"action": "update", "timestamp": 1700000050}
    ]
}

print("--- 1. 基础用法 (rich.print) ---")
rprint(complex_data)


print("\n--- 2. JSON 专用打印 (rich.print_json) ---")
console = Console()
console.print_json(data=complex_data)


print("\n--- 3. 终极完美：带边框和标题 (Panel + Pretty) ---")
pretty_data = Pretty(complex_data)
panel = Panel(
    pretty_data, 
    title="[bold blue]User Profile[/]", 
    subtitle="[italic]Fetched from DB[/]",
    border_style="green",
    expand=False
)
rprint(panel)


print("\n--- 4. 另一种风格：语法高亮面板 (Panel + Syntax) ---")
json_str = json.dumps(complex_data, indent=2, default=str)
syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)

rprint(Panel(
    syntax, 
    title="JSON View", 
    border_style="magenta"
))
