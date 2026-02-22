import ast
from typing import List, Dict

class DeadCodeDetector(ast.NodeVisitor):
    """
    A deterministic static analysis tool to detect unreachable code.
    Operates by sequentially scanning Basic Blocks within the AST.
    """
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.unreachable_nodes: List[ast.AST] = []
        # Routing nodes that disrupt linear control flow
        self.terminal_types = (ast.Return, ast.Raise, ast.Break, ast.Continue)
        
        try:
            self.tree = ast.parse(source_code)
        except SyntaxError as e:
            raise SyntaxError(f"Internal syntax error. Code cannot be compiled: {e}")

    def analyze(self) -> List[Dict]:
        """Executes the analysis and returns metadata of dead code nodes."""
        self.visit(self.tree)
        return [
            {
                "line": node.lineno,
                "col_offset": node.col_offset,
                "type": type(node).__name__
            }
            for node in self.unreachable_nodes
        ]

    def visit(self, node: ast.AST):
        """Overrides NodeVisitor to intercept blocks containing a body."""
        if hasattr(node, 'body') and isinstance(node.body, list):
            self._check_block(node.body)
        
        if hasattr(node, 'orelse') and isinstance(node.orelse, list):
            self._check_block(node.orelse)
            
        # Continue deep traversal into child branches
        self.generic_visit(node)

    def _check_block(self, block_nodes: List[ast.AST]):
        """Linearly traverses a code block to detect dead branches."""
        terminal_found = False
        for child in block_nodes:
            if terminal_found:
                self.unreachable_nodes.append(child)
            elif isinstance(child, self.terminal_types):
                terminal_found = True