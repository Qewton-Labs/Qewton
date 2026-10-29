class RenderApplication:
    """Base class for wrapping a Figure in an interactive web application."""

    @staticmethod
    def create(figure):
        """Builds and returns an application serving `figure`, with an
        interactive widget for each control it connects."""
        raise NotImplementedError
