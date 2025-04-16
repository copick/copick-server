# As a module
from copick_server.server import serve_copick

serve_copick("example_copick.json", allowed_origins=["*"])

# Or from command line
# $ python server.py path/to/copick_config.json --cors "*"
