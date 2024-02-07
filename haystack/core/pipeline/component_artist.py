import numpy as np
from matplotlib.artist import Artist
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
from matplotlib.transforms import Bbox, IdentityTransform


class ComponentArtist(Artist):
    def __init__(self, x: float, y: float, name: str, type_name: str):
        super().__init__()
        # self.padding = 0.1
        self._x = x
        self._y = y

        self._name = Text(x, y, text=name, horizontalalignment="center", color="black", bbox={"pad": 0})

        self._type_name = Text(
            x,
            y,
            text=type_name,
            horizontalalignment="center",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.5, "pad": 0},
        )
        self._padding = 5
        self._box = FancyBboxPatch((0, 0), 0, 0, facecolor="white")

    def get_bbox(self) -> Bbox:
        return self._box.get_bbox()

    def axes(self, new_axes):
        self._name.axes = new_axes
        self._type_name.axes = new_axes

    def set_transform(self, transform):
        super().set_transform(transform)
        self._name.set_transform(transform)
        self._type_name.set_transform(transform)
        # self._box.set_transform(transform)

    def set_figure(self, fig):
        super().set_figure(fig)
        self._name.set_figure(fig)
        self._type_name.set_figure(fig)

    def draw(self, renderer):
        renderer_group = "ComponentArtist"
        renderer.open_group(renderer_group, self.get_gid())

        # Calculate Component name position
        name_bbox, name_info, _ = self._name._get_layout(renderer)
        trans = self._name.get_transform()
        # name_posx = float(self.convert_xunits(self._name._x))
        # name_posy = float(self.convert_yunits(self._name._y))
        name_posx = self._name._x
        name_posy = self._name._y
        name_posx, name_posy = trans.transform((name_posx, name_posy))
        name_posy += name_bbox.height / 2
        if not np.isfinite(name_posx) or not np.isfinite(name_posy):
            return

        # Calculate Component type name position
        type_name_bbox, type_name_info, _ = self._type_name._get_layout(renderer)
        trans = self._type_name.get_transform()
        # type_name_posx = float(self.convert_xunits(self._type_name._x))
        # type_name_posy = float(self.convert_yunits(self._type_name._y))
        type_name_posx = self._type_name._x
        type_name_posy = self._type_name._y

        type_name_posx, type_name_posy = trans.transform((type_name_posx, type_name_posy))
        type_name_posy -= type_name_bbox.height / 2
        if not np.isfinite(type_name_posx) or not np.isfinite(type_name_posy):
            return

        # Get the maximum width and X position to properly draw the box
        max_x = max(name_posx, type_name_posx)
        max_width = max(name_bbox.width, type_name_bbox.width)

        # Update box position and size and draw it
        self._box.set_x(max_x - max_width / 2 - self._padding)
        self._box.set_y(type_name_posy - self._padding)
        self._box.set_width(max_width + self._padding * 2)
        self._box.set_height(name_bbox.height + type_name_bbox.height + self._padding * 2)
        self._box.draw(renderer)

        gc = renderer.new_gc()

        angle = 0
        _, canvash = renderer.get_canvas_width_height()

        # Draw the Component name
        for line, _, x, y in name_info:
            mtext = self if len(name_info) == 1 else None
            x = x + name_posx
            y = y + name_posy
            if renderer.flipy():
                y = canvash - y
            clean_line, ismath = self._name._preprocess_math(line)
            renderer.draw_text(gc, x, y, clean_line, self._name._fontproperties, angle, ismath=ismath, mtext=mtext)

        # Draw the Component type name
        for line, _, x, y in type_name_info:
            mtext = self if len(type_name_info) == 1 else None
            x = x + type_name_posx
            y = y + type_name_posy
            if renderer.flipy():
                y = canvash - y
            clean_line, ismath = self._type_name._preprocess_math(line)
            renderer.draw_text(gc, x, y, clean_line, self._type_name._fontproperties, angle, ismath=ismath, mtext=mtext)

        gc.restore()
        renderer.close_group(renderer_group)
        self.stale = False
