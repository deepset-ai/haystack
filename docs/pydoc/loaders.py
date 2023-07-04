import copy
import typing as t

import docspec
from pydoc_markdown.interfaces import Context
from pydoc_markdown.contrib.loaders.python import PythonLoader


class CustomPythonLoader(PythonLoader):
    def load(self) -> t.Iterable[docspec.Module]:
        """
        Load the modules, but include inherited methods in the classes.
        """
        # Load all haystack modules
        temp_loader = PythonLoader(search_path=["../../../haystack"])
        temp_loader.init(Context(directory="."))
        all_modules = list(temp_loader.load())

        # Collect all classes
        classes = {}
        for module in all_modules:
            for member in module.members:
                if isinstance(member, docspec.Class):
                    classes[member.name] = member

        # Load the modules specified in the search path
        modules = super().load()

        # Add inherited methods to the classes
        modules = self.include_inherited_methods(modules, classes)

        return modules

    def include_inherited_methods(
        self, modules: t.Iterable[docspec.Module], classes: t.Dict[str, docspec.Class]
    ) -> t.Iterable[docspec.Module]:
        """
        Recursively include inherited methods from the base classes.
        """
        modules = list(modules)
        for module in modules:
            for cls in module.members:
                if isinstance(cls, docspec.Class):
                    self.include_methods_for_class(cls, classes)

        return modules

    def include_methods_for_class(self, cls: docspec.Class, classes: t.Dict[str, docspec.Class]):
        """
        Include all methods inherited from base classes to the class.
        """
        if cls.bases is None:
            return
        for base in cls.bases:
            if base in classes:
                base_cls = classes[base]
                self.include_methods_for_class(base_cls, classes)

                for member in base_cls.members:
                    if isinstance(member, docspec.Function) and not any(m.name == member.name for m in cls.members):
                        new_member = copy.deepcopy(member)
                        new_member.parent = cls
                        cls.members.append(new_member)
