import importlib
import json
import os
from pprint import pprint
import shutil
import sys
import textwrap

import click

from util import paths_relative_to
from util import paths_within

import procurement


relpath = paths_relative_to(__file__)


CONTEXT_FILE = relpath("context.json")

REPO_DIR = relpath("repos")

if not os.path.exists(REPO_DIR):
    os.makedirs(REPO_DIR)

in_repo_dir = paths_within(REPO_DIR)


def registry_from_context(context_name):
    context_dir = in_repo_dir(context_name)
    if os.path.exists(context_dir):
        sys.path.append(context_dir)
        registry = importlib.import_module("register")
        return registry.registry
    else:
        raise ValueError(
            f"No known procurement context {context_name}."
        )


@click.group()
def cli():
    """Traverse complicated procurement trees."""


@cli.command()
@click.option("-n", "--context-name", prompt=True)
def init(context_name):
    """Initialize a new procurement context."""
    context_dir = in_repo_dir(context_name)
    if not os.path.exists(context_dir):
        os.makedirs(context_dir)
    else:
        raise ValueError(
            f"Procurement context {context_name} already exists!"
        )

    in_context_dir = paths_within(context_dir)

    with open(in_context_dir("recipes.txt"), "w") as f:
        f.write("")

    init_py = textwrap.dedent("""
    from procurement import Procurement
    from procurement import Buy
    from procurement import Cost
    from procurement import Costs
    from procurement import Craft
    from procurement import CraftHeterogeneous
    from procurement import CraftHomogeneous
    from procurement import Gather
    from procurement import Ingredients
    from procurement import Registry

    from util import all_subclasses
    from util import procure_alias
    from util import paths_relative_to


    relpath = paths_relative_to(__file__)


    registry = Registry(
        kind=Ingredients,
        default_procurement=Craft,
        procurements={
            c.procurement_name(): c for c in all_subclasses(Procurement)
        },
    )


    with open(relpath("recipes.txt")) as f:
        registry.from_lines(f)
    """)

    with open(in_context_dir("register.py"), "w") as f:
        f.writelines(init_py)

    click.echo(f"Created context {context_name}")

    if os.path.exists(in_repo_dir(context_name)):
        with open(CONTEXT_FILE, "w") as f:
            json.dump({"context": context_name}, f)
    else:
        raise RuntimeError(
            f"Provided procurement context {context_name} does not exist!  "
            f"Use `procure list` to see available contexts."
        )


@cli.command()
def where():
    """Emit the location of the procurement contexts."""
    click.echo(REPO_DIR)


@cli.command()
def list():
    """List known procurement contexts."""
    for line in os.listdir(REPO_DIR):
        click.echo(line)


@cli.command()
@click.argument("context_name")
def remove(context_name):
    """
    Remove a procurement context.

    This cannot be undone!
    """
    if click.confirm(
        f"Are you sure you want to remove context {context_name}?  This "
        f"cannot be undone!"
    ):
        shutil.rmtree(in_repo_dir(context_name))
        if os.path.exists(CONTEXT_FILE):
            os.unlink(CONTEXT_FILE)
    else:
        click.echo("Removal not confirmed, aborting.")


@cli.command()
@click.argument("context_name")
def use(context_name):
    """Activate a known procurement context."""
    if os.path.exists(in_repo_dir(context_name)):
        with open(CONTEXT_FILE, "w") as f:
            json.dump({"context": context_name}, f)
    else:
        raise RuntimeError(
            f"Provided procurement context {context_name} does not exist!  "
            f"Use `procure list` to see available contexts."
        )


def current_context(fail=True):
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, "r") as f:
            data = json.load(f)
        context = data.get("context")
    else:
        context = None

    if fail and context is None:
        raise LookupError("No current context")
    else:
        return context


@cli.command()
def current():
    """Emit the current procurement context."""
    context = current_context()

    if context:
        click.echo(context)
    else:
        click.echo("No current context.  Enter one with `procure use`")


@cli.command()
def quit():
    """Exit the procurement context."""
    try:
        os.unlink(CONTEXT_FILE)
        if found:
            click.echo(f"Quit context {found}")
        else:
            click.echo("No context to quit")
    except FileNotFoundError:
        click.echo("No context to quit")


@cli.command()
@click.option("-n", "--context-name", default=None)
@click.option("--recipe", "--recipes", is_flag=True, default=False)
@click.option("--registry", "--register", is_flag=True, default=False)
def edit(context_name=None, recipe=False, registry=False):
    """Edit the recipes for the context."""
    found = context_name or current_context()

    if found:
        context_dir = in_repo_dir(found)
        in_context_dir = paths_within(context_dir)

        if recipe:
            which_file = "recipes.txt"
        elif registry:
            which_file = "register.py"
        else:
            which_file = "recipes.txt"

        with open(in_context_dir(which_file)) as f:
            contents = f.read()
        edited = click.edit(contents)
        if edited is not None and edited != contents:
            with open(in_context_dir(which_file), "w") as f:
                f.write(edited)
        else:
            click.echo("No changes")
    else:
        click.echo("No context to edit")


@cli.command()
@click.option("-n", "--context-name", default=None)
@click.argument("recipe_string")
def optimize(context_name, recipe_string):
    """Emit the depenency tree with details."""
    found = context_name or current_context()
    registry = registry_from_context(found)
    result = procurement.optimize(
        registry,
        registry.kind.parse(recipe_string),
    )
    pprint(result)


@cli.command()
@click.option("-n", "--context-name", default=None)
@click.argument("recipe_string")
def dnf(context_name, recipe_string):
    """
    Emit the disjunctive normal form of the tree.

    Use this with caution, this output can be very large.
    """
    found = context_name or current_context()
    registry = registry_from_context(found)
    tree = procurement.optimize(
        registry,
        registry.kind.parse(recipe_string),
    )
    result = procurement.dnf(tree)
    for entry in result:
        pprint(procurement.join_opt_results(entry))


@cli.command()
@click.option("-n", "--context-name", default=None)
@click.argument("recipe_string")
def summary(context_name, recipe_string):
    """Emit a summary of the dependency tree without details."""
    found = context_name or current_context()
    registry = registry_from_context(found)
    tree = procurement.optimize(registry, registry.kind.parse(recipe_string))
    result = procurement.process_tree_overview(tree, pretty=True)
    print(result)


@cli.command()
@click.option("-n", "--context-name", default=None)
def recipes(context_name):
    """Emit the products with known recipes."""
    found = context_name or current_context()
    registry = registry_from_context(found)
    for (k, v) in registry.registry.items():
        print(f"{k} ({len(v)} known)")


if __name__ == "__main__":
    cli()
