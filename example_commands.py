class ExampleCommands:
    def __init__(self, io, coder):
        self.io = io
        self.coder = coder

    def cmd_index(self, args):
        """Create or update an index of the codebase
        Usage: /index [path]
        If no path provided, indexes current working directory"""
        try:
            path = args.strip() or "."
            files = self._get_files_to_index(path)
            self.io.tool_output(f"Indexed {len(files)} files:\n" + "\n".join(files))
            return files
        except Exception as e:
            self.io.tool_error(f"Indexing failed: {e}")
            return []

    def _get_files_to_index(self, path):
        """Helper method to find all code files in a directory"""
        # Implement actual file discovery logic here
        # This is just a placeholder implementation
        import os
        if os.path.isfile(path):
            return [path]
        return [
            os.path.join(root, f)
            for root, _, files in os.walk(path)
            for f in files
            if f.endswith(('.py', '.js', '.html', '.css'))  # Add other extensions as needed
        ]
