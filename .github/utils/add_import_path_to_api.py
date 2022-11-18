import haystack
import inspect
import os

ROOT = "haystack"

ignore_modules = ["logging", "config", "__config__", "np"]
markdowns_dir = "docs/_src/api/api/temp"


def add_import_path(import_paths):
    """Add a line under the Class API file with the import path for the class"""
    markdown_files = [x for x in os.listdir(markdowns_dir) if x[-3:] == ".md"]
    for mdf in markdown_files:
        lines_new = []
        with open(os.path.join(markdowns_dir, mdf), "r") as f:
            for l in f:
                l_new = l
                if l[:2] == "## ":
                    classname = l[3:].strip()
                    if classname in import_paths:
                        l_new = l + "{}".format(import_paths[classname]) + "\n"
                lines_new.append(l_new)
        with open(os.path.join(markdowns_dir, mdf), "w") as f:
            f.writelines(lines_new)


def generate_shortest_class_imports(python_package):
    """Recursively go through all modules and classes in a package and find the shortest import path for each class"""
    shortest_map = {}
    curr_path = ROOT
    members_graph = inspect.getmembers(haystack)
    depth = 0
    return generate_shortest_class_imports_recursive(python_package, shortest_map, members_graph, curr_path, depth)


def generate_shortest_class_imports_recursive(python_package, shortest_map, members_graph, curr_path, depth):
    """Recursively go through all modules and classes in a package and find the shortest import path for each class"""
    if depth > 2:
        return shortest_map
    for name, member in members_graph:
        if inspect.isclass(member):
            add_shortest_path(curr_path, name, shortest_map)
        elif inspect.ismodule(member):
            try:
                members_graph = inspect.getmembers(member)
                generate_shortest_class_imports_recursive(
                    python_package, shortest_map, members_graph, curr_path + "." + name, depth + 1
                )
            except:
                pass
    return shortest_map


def add_shortest_path(curr_path, class_name, shortest_map):
    """Add the shortest import path for a class to the shortest_map provided it is the shortest path"""
    if class_name not in shortest_map:
        shortest_map[class_name] = curr_path + "." + class_name
    elif len(curr_path.split(".")) < len(shortest_map[class_name].split(".")):
        shortest_map[class_name] = curr_path + "." + class_name


def relevant_class(path):
    """Returns True if the class is only one level deep in the module (e.g. haystack.Document) or if it is from a relevant submodule."""
    if len(path.split(".")) == 2:
        return True
    elif relevant_module(path):
        return True
    return False


def relevant_module(path):
    """Returns True if the path is from a relevant submodule."""
    relevant_haystack_modules = ["pipelines", "nodes", "utils", "document_stores"]
    for relevant_module in relevant_haystack_modules:
        if relevant_module in path:
            return True
    return False


if __name__ == "__main__":
    shortest_map = generate_shortest_class_imports(haystack)
    filter_shortest_map = {k: v for k, v in shortest_map.items() if relevant_class(v)}
    add_import_path(filter_shortest_map)
