"""Base class of the applications, which show a Figure as a connected
output."""

import importlib.util

from qewton.visualization.axes import Control


class RenderApplication:
    """Base class for wrapping a Figure in an interactive application: a
    connected output, in which widgets set the states of controls.

    Attributes:
        connected (tuple of type): Kinds of control that get a widget
            (class attribute); the other controls are embedded into the
            drawn figure.

    Notes:
        Subclasses set :attr:`connected` and implement
        :meth:`renderer_classes` and :meth:`create`.
    """

    connected: tuple[type, ...] = ()

    @classmethod
    def renderer_classes(cls) -> tuple[type, ...]:
        """The renderers whose figures the application shows.

        Returns:
            tuple of type: Renderer classes.
        """
        raise NotImplementedError

    @classmethod
    def create(cls, figure):
        """Build an application serving `figure`, with a widget for each
        connected control.

        Args:
            figure (Figure): The figure to serve.

        Returns:
            object: The application.
        """
        raise NotImplementedError

    @classmethod
    def check_renderer(cls, figure) -> None:
        """Fail early if the application cannot show `figure`.

        Args:
            figure (Figure): The figure.

        Raises:
            TypeError: If the figure's renderer is none of
                :meth:`renderer_classes`.
        """
        supported = cls.renderer_classes()
        if not isinstance(figure.renderer, supported):
            names = " or ".join(r.__name__ for r in supported)
            raise TypeError(
                f"{cls.__name__} shows figures drawn by a {names}, not by a "
                f"{type(figure.renderer).__name__}."
            )

    @classmethod
    def interactive_controls(cls, figure) -> list[Control]:
        """The controls of `figure` that get a widget, in widget order.

        Args:
            figure (Figure): The figure.

        Returns:
            list of Control: Its controls of a :attr:`connected` kind.
        """
        return [c for c in figure.controls if type(c) in cls.connected]

    @classmethod
    def embedded_controls(cls, figure) -> list[Control]:
        """The controls of `figure` without a widget, embedded into the
        drawn figure.

        Args:
            figure (Figure): The figure.

        Returns:
            list of Control: Every other control.
        """
        connected = cls.interactive_controls(figure)
        return [c for c in figure.controls if c not in connected]

    @classmethod
    def update(cls, figure, controls: list[Control], states) -> object:
        """Set the states of `controls` and draw `figure` with the other
        controls embedded, as a widget callback does.

        Args:
            figure (Figure): The figure.
            controls: Its interactive controls.
            states: One state per control.

        Returns:
            object: The redrawn backend figure.
        """
        for control, state in zip(controls, states):
            control.state = int(state)
        return figure.draw(embedded=cls.embedded_controls(figure))


def require(application: str, *packages: str) -> None:
    """Fail early if optional packages of an application are missing.

    Args:
        application: Name of the application, for the message.
        *packages: Importable package names.

    Raises:
        ImportError: Naming every missing package.
    """
    missing = [p for p in packages if importlib.util.find_spec(p) is None]
    if missing:
        raise ImportError(
            f"{application} needs {', '.join(missing)}; install with "
            f"`pip install {' '.join(missing)}`."
        )
