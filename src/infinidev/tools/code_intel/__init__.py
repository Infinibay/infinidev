"""Code intelligence tools for the agent."""

from infinidev.tools.code_intel.find_definition import FindDefinitionTool
from infinidev.tools.code_intel.find_references import FindReferencesTool
from infinidev.tools.code_intel.list_symbols import ListSymbolsTool
from infinidev.tools.code_intel.search_symbols import SearchSymbolsTool
from infinidev.tools.code_intel.get_symbol_code import GetSymbolCodeTool
from infinidev.tools.code_intel.project_structure import ProjectStructureTool
from infinidev.tools.code_intel.edit_method import EditSymbolTool
from infinidev.tools.code_intel.add_method import AddSymbolTool
from infinidev.tools.code_intel.remove_method import RemoveSymbolTool
from infinidev.tools.code_intel.analyze_code import AnalyzeCodeTool
from infinidev.tools.code_intel.rename_symbol import RenameSymbolTool
from infinidev.tools.code_intel.move_symbol import MoveSymbolTool
from infinidev.tools.code_intel.find_similar_methods_tool import FindSimilarMethodsTool

# Backward-compat aliases
EditMethodTool = EditSymbolTool
AddMethodTool = AddSymbolTool
RemoveMethodTool = RemoveSymbolTool

__all__ = [
    "FindDefinitionTool", "FindReferencesTool",
    "ListSymbolsTool", "SearchSymbolsTool", "GetSymbolCodeTool",
    "ProjectStructureTool",
    "EditSymbolTool", "AddSymbolTool", "RemoveSymbolTool",
    "EditMethodTool", "AddMethodTool", "RemoveMethodTool",  # aliases
    "AnalyzeCodeTool",
    "RenameSymbolTool", "MoveSymbolTool",
    "FindSimilarMethodsTool",
]
