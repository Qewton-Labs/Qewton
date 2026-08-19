class Renderer:
    @staticmethod
    def setup(figure):
        raise NotImplementedError()

    @staticmethod
    def animate(figure, backend_figure, spec):
        """Materializes one frame per `spec.values` on top of an already-drawn
        backend_figure, plus whatever play/pause/scrub UI this backend uses.
        Only called when a TimeSpec is present (Figure.draw()) - backends
        without a frame-based animation model can leave this unimplemented,
        same as an unsupported feature raising here rather than silently
        doing nothing."""
        raise NotImplementedError()

    @staticmethod
    def show(backend_figure):
        raise NotImplementedError()

    @staticmethod
    def save_html(backend_figure, path):
        raise NotImplementedError()

    @staticmethod
    def save_gif(backend_figure, path, fps=10):
        """Renders each animation frame (see animate()) to a static image and
        assembles them into a looping GIF. Requires backend_figure to already
        carry frames, i.e. Figure.draw() must have run with a TimeSpec
        present. Backends implementing this may depend on an optional
        rasterization library (e.g. Plotly needs 'kaleido') - imported lazily
        inside the implementation, not a hard dependency of this package,
        same convention as HDF5DataSet.from_file()'s h5py import."""
        raise NotImplementedError()


class Artist:
    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        raise NotImplementedError

    def update(self, backend_figure, plot):
        raise NotImplementedError

    def remove(self, backend_figure):
        raise NotImplementedError
