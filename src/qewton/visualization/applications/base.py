class RenderApplication:
    """Base class for wrapping a Figure in an interactive web application."""

    @staticmethod
    def create(figure):
        """Builds and returns an application serving `figure`, wiring up
        an interactive widget for each SliderSpec control it declares."""
        raise NotImplementedError
