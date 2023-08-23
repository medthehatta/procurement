import importlib
import json
import os
import shutil
import sys
import textwrap

import click

from util import paths_relative_to
from util import paths_within


relpath = paths_relative_to(__file__)


context_file = relpath("context.json")

repo_dir = relpath("repos")

if not os.path.exists(repo_dir):
    os.makedirs(repo_dir)

in_repo_dir = paths_within(repo_dir)


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
@click.option("-n", "--context_name", prompt=True)
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
    from procurement import Buy
    from procurement import Cost
    from procurement import Costs
    from procurement import Craft
    from procurement import CraftHeterogeneous
    from procurement import CraftHomogeneous
    from procurement import Gather
    from procurement import Ingredients
    from procurement import Registry

    from util import procure_alias
    from util import paths_relative_to


    relpath = paths_relative_to(__file__)


    registry = Registry(
        kind=Ingredients,
        default_procurement=Craft,
        procurements={},
    )


    with open(relpath("recipes.txt")) as f:
        registry.from_lines(f)
    """)

    with open(in_context_dir("register.py"), "w") as f:
        f.writelines(init_py)


@cli.command()
def where():
    """Emit the location of the procurement contexts."""
    click.echo(repo_dir)


@cli.command()
def list():
    """List known procurement contexts."""
    for line in os.listdir(repo_dir):
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
    else:
        click.echo("Removal not confirmed, aborting.")


@cli.command()
@click.argument("context_name")
def use(context_name):
    """Activate a known procurement context."""
    if os.path.exists(in_repo_dir(context_name)):
        with open(context_file, "w") as f:
            json.dump({"context": context_name}, f)
    else:
        raise RuntimeError(
            f"Provided procurement context {context_name} does not exist!  "
            f"Use `procure list` to see available contexts."
        )


@cli.command()
def current():
    """Emit the current procurement context."""
    data = None

    if os.path.exists(context_file):
        with open(context_file, "r") as f:
            data = json.load(f)

    if data:
        click.echo(data.get("context"))
    else:
        click.echo("No current context.  Enter one with `procure use`")


@cli.command()
def quit():
    """Exit the procurement context."""
    data = None

    if os.path.exists(context_file):
        with open(context_file, "r") as f:
            data = json.load(f)

    if data:
        found = data.get("context")
    else:
        found = None

    try:
        os.unlink(context_file)
        if found:
            click.echo(f"Quit context {found}")
        else:
            click.echo("No context to quit")
    except FileNotFoundError:
        click.echo("No context to quit")


if __name__ == "__main__":
    cli()
