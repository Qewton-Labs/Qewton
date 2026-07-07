class Renderer:
    @staticmethod
    def setup():
        raise NotImplementedError()

    @staticmethod
    def show(backend_figure):
        raise NotImplementedError()

    @staticmethod
    def save_html(backend_figure, path):
        raise NotImplementedError()


class Artist:
    @classmethod
    def create(cls, backend_figure, plot):
        raise NotImplementedError

    def update(self, backend_figure, plot):
        raise NotImplementedError

    def remove(self, backend_figure):
        raise NotImplementedError
