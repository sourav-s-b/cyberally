from blue.training import harness_bc_redsuite as suite
import ast
import inspect

from CybORG.Agents.SimpleAgents.FSMRedVariants import VerboseFSRed
from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent


def test_red_suite_and_seed_design_is_frozen_and_disjoint():
    assert suite.REDS == ("discovery", "verbose", "random", "sleep")
    assert len(suite.SEEDS) == 32 and len(set(suite.SEEDS)) == 32
    assert not set(suite.SEEDS) & set(range(8253, 8285))
    assert len(suite.MODEL_SEEDS) == 5


def test_verbose_variant_only_changes_print_flags():
    variant = ast.parse(inspect.getsource(VerboseFSRed))
    cls = next(n for n in variant.body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert [m.name for m in methods] == ["__init__"]
    assigned = {n.targets[0].attr for n in ast.walk(methods[0])
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute)
                and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "self"}
    assert assigned == {"print_action_output", "print_obs_output"}

    base = ast.parse(inspect.getsource(FiniteStateRedAgent))
    guarded = {n.test.attr for n in ast.walk(base)
               if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
               and isinstance(n.test.value, ast.Name) and n.test.value.id == "self"}
    assert {"print_action_output", "print_obs_output"} <= guarded
